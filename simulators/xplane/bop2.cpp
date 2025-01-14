#include <stdio.h>
#include <stdlib.h>
#include "../inc/prefetcher.h"

// Submission ID: 3

// Paper title: A Best-Offset Prefetcher

// Author: Pierre Michaud


//######################################################################################
//                             PREFETCHER PARAMETERS
//######################################################################################

// Because prefetch cannot cross 4KB-page boundaries, there is no need to consider offsets
// greater than 63. However, with pages larger than 4KB, it would be beneficial to consider
// larger offsets.

#define NOFFSETS 46
int OFFSET[NOFFSETS] = {1,-1,2,-2,3,-3,4,-4,5,-5,6,-6,7,-7,8,-8,9,-9,10,-10,11,-11,12,-12,13,-13,14,-14,15,-15,16,-16,18,-18,20,-20,24,-24,30,-30,32,-32,36,-36,40,-40};
#define DEFAULT_OFFSET 1
#define SCORE_MAX 31
#define ROUND_MAX 100
#define RRTAG 12
#define DELAYQSIZE 15
#define DELAY 60
#define TIME_BITS 12
#define LLC_RATE_MAX 255
#define GAUGE_MAX 8191
#define MSHR_THRESHOLD_MAX (L2_MSHR_COUNT-4)  // TODO: find these
#define MSHR_THRESHOLD_MIN 2                  // TODO: find these
#define LOW_SCORE 20
#define BAD_SCORE ((knob_small_llc)? 10 : 1)
#define BANDWIDTH ((knob_low_bandwidth)? 64 : 16)




//######################################################################################
//                            SOME MACROS & DEFINITIONS
//######################################################################################

#define LOGLINE 6

#define SAMEPAGE(lineaddr1,lineaddr2) ((((lineaddr1) ^ (lineaddr2)) >> 6) == 0)

#define INCREMENT(x,n) {x++; if (x==(n)) x=0;}

#define TRUNCATE(x,nbits) (((x) & ((1<<(nbits))-1)))





//######################################################################################
//                            RECENT REQUESTS TABLE (RR)
//######################################################################################

void BestOffsetPrefetcher::rr_init()
{
  int i;
  for (i=0; i<(1<<RRINDEX); i++) {
    recent_request[0][i] = 0;
    recent_request[1][i] = 0;
  }
}


int BestOffsetPrefetcher::rr_tag(t_addr lineaddr)
{
  return TRUNCATE(lineaddr>>RRINDEX,RRTAG);
}


int BestOffsetPrefetcher::rr_index_left(t_addr lineaddr)
{
  return TRUNCATE(lineaddr^(lineaddr>>RRINDEX),RRINDEX);
}


int BestOffsetPrefetcher::rr_index_right(t_addr lineaddr)
{
  return TRUNCATE(lineaddr^(lineaddr>>(2*RRINDEX)),RRINDEX);
}


void BestOffsetPrefetcher::rr_insert_left(t_addr lineaddr)
{
  int i = rr_index_left(lineaddr);
  recent_request[0][i] = rr_tag(lineaddr);
}


void BestOffsetPrefetcher::rr_insert_right(t_addr lineaddr)
{
  int i = rr_index_right(lineaddr);
  recent_request[1][i] = rr_tag(lineaddr);
}


int BestOffsetPrefetcher::rr_hit(t_addr lineaddr)
{
  int i = rr_index_left(lineaddr);
  int j = rr_index_right(lineaddr);
  int tag = rr_tag(lineaddr);
  return (recent_request[0][i] == tag) || (recent_request[1][j] == tag);
}



//######################################################################################
//                               DELAY QUEUE (DQ)
//######################################################################################

// Without the delay queue, the prefetcher would always try to select an offset value
// large enough for having timely prefetches. However, sometimes, a small offset yields
// late prefetches but greater prefetch accuracy and better performance. The delay queue
// is an imperfect solution to this problem.

// This implementation of the delay queue is specific to the DPC2 simulator, as the DPC2
// prefetcher can act only at certain clock cycles. In a real processor, the delay queue
// implementation can be simpler.


void BestOffsetPrefetcher::dq_init()
{
  int i;
  for (i=0; i<DELAYQSIZE; i++) {
    dq.lineaddr[i] = 0;
    dq.cycle[i] = 0;
    dq.valid[i] = 0;
  }
  dq.tail = 0;
  dq.head = 0;
}


void BestOffsetPrefetcher::dq_push(t_addr lineaddr)
{
  // enqueue one line address
  if (dq.valid[dq.tail]) {
    // delay queue is full
    // dequeue the oldest entry and write the "left" bank of the RR table
    rr_insert_left(dq.lineaddr[dq.head]);
    INCREMENT(dq.head,DELAYQSIZE);
  }
  dq.lineaddr[dq.tail] = TRUNCATE(lineaddr,RRINDEX+RRTAG);
  dq.cycle[dq.tail] = TRUNCATE(get_current_cycle(0),TIME_BITS);
  dq.valid[dq.tail] = 1;
  INCREMENT(dq.tail,DELAYQSIZE);
}


int BestOffsetPrefetcher::dq_ready()
{
  // tells whether or not the oldest entry is ready to be dequeued
  if (! dq.valid[dq.head]) {
    // delay queue is empty
    return 0;
  }
  int cycle = TRUNCATE(get_current_cycle(0),TIME_BITS);
  int issuecycle = dq.cycle[dq.head];
  int readycycle = TRUNCATE(issuecycle+DELAY,TIME_BITS);
  if (readycycle >= issuecycle) {
    return (cycle < issuecycle) || (cycle >= readycycle);
  } else {
    return (cycle < issuecycle) && (cycle >= readycycle);
  }
}


void BestOffsetPrefetcher::dq_pop()
{
  // dequeue the entries that are ready to be dequeued,
  // and do a write in the "left" bank of the RR table for each of them
  int i;
  for (i=0; i<DELAYQSIZE; i++) {
    if (! dq_ready()) {
      break;
    }
    rr_insert_left(dq.lineaddr[dq.head]);
    dq.valid[dq.head] = 0;
    INCREMENT(dq.head,DELAYQSIZE);
  }
}



//######################################################################################
//                               PREFETCH THROTTLE (PT)
//######################################################################################

// The following prefetch throttling method is specific to the DPC2 simulator, as other
// parts of the microarchitecture (requests schedulers, cache replacement policy,
// LLC hit/miss information,...) can be neither modified nor observed. Consequently,
// we ignore hardware implementation considerations here.


void BestOffsetPrefetcher::pt_init()
{
  pt.mshr_threshold = MSHR_THRESHOLD_MAX;
  pt.prefetch_score = SCORE_MAX;
  pt.llc_rate = 0;
  pt.llc_rate_gauge = GAUGE_MAX/2;
  pt.last_cycle = 0;
}


// The pt_update_mshr_threshold function is for adjusting the MSHR threshold
// (a prefetch request is dropped when the MSHR occupancy exceeds the threshold)

void BestOffsetPrefetcher::pt_update_mshr_threshold()
{
  if ((pt.prefetch_score > LOW_SCORE) || (pt.llc_rate > (2*BANDWIDTH))) {
    // prefetch accuracy not too bad, or low bandwidth requirement
    // ==> maximum prefetch aggressiveness
    pt.mshr_threshold = MSHR_THRESHOLD_MAX;
  } else if (pt.llc_rate < BANDWIDTH) {
    // LLC access rate exceeds memory bandwidth, implying that there are some LLC hits.
    // If there are more LLC misses than hits, perhaps memory bandwidth saturates.
    // If there are more LLC hits than misses, the MSHR is probably not stressed.
    // So we set the MSHR threshold low.
    pt.mshr_threshold = MSHR_THRESHOLD_MIN;
  } else {
    // in-between situation: we set the MSHR threshold proportionally to the (inverse) LLC rate
    pt.mshr_threshold = MSHR_THRESHOLD_MIN + (MSHR_THRESHOLD_MAX-MSHR_THRESHOLD_MIN) * (double) (pt.llc_rate - BANDWIDTH) / BANDWIDTH;
  }
}


// The pt_llc_access function estimates the average time between consecutive LLC accesses.
// It is called on every LLC access.

void BestOffsetPrefetcher::pt_llc_access()
{
  // update the gauge
  int cycle = TRUNCATE(get_current_cycle(0),TIME_BITS);
  int dt = TRUNCATE(cycle - pt.last_cycle,TIME_BITS);
  pt.last_cycle = cycle;
  pt.llc_rate_gauge += dt - pt.llc_rate;

  // if the gauge reaches its upper limit, increment the rate counter
  // if the gauge reaches its lower limit, decrement the rate counter
  // otherwise leave the rate counter unchanged
  if (pt.llc_rate_gauge > GAUGE_MAX) {
    pt.llc_rate_gauge = GAUGE_MAX;
    if (pt.llc_rate < LLC_RATE_MAX) {
      pt.llc_rate++;
      pt_update_mshr_threshold();
    }
  } else if (pt.llc_rate_gauge < 0) {
    pt.llc_rate_gauge = 0;
    if (pt.llc_rate > 0) {
      pt.llc_rate--;
      pt_update_mshr_threshold();
    }
  }
}


//######################################################################################
//                               OFFSETS SCORES (OS)
//######################################################################################

// A method for determining the best offset value

void BestOffsetPrefetcher::os_reset()
{
  int i;
  for (i=0; i<NOFFSETS; i++) {
    os.score[i] = 0;
  }
  os.max_score = 0;
  os.best_offset = 0;
  os.round = 0;
  os.p = 0;
}


// The os_learn_best_offset function tests one offset at a time, trying to determine
// if the current line would have been successfully prefetched with that offset

void BestOffsetPrefetcher::os_learn_best_offset(t_addr lineaddr)
{
  int testoffset = OFFSET[os.p];
  t_addr testlineaddr = lineaddr - testoffset;

  if (SAMEPAGE(lineaddr,testlineaddr) && rr_hit(testlineaddr)) {
    // the current line would likely have been prefetched successfully with that offset
    // ==> increment the score
    os.score[os.p]++;
    if (os.score[os.p] >= os.max_score) {
      os.max_score = os.score[os.p];
      os.best_offset = testoffset;
    }
  }

  if (os.p == (NOFFSETS-1)) {
    // one round finished
    os.round++;

    if ((os.max_score == SCORE_MAX) || (os.round == ROUND_MAX)) {
      // learning phase is finished, update the prefetch offset
      prefetch_offset = (os.best_offset != 0)? os.best_offset : DEFAULT_OFFSET;
      pt.prefetch_score = os.max_score;
      pt_update_mshr_threshold();

      if (os.max_score <= BAD_SCORE) {
	// prefetch accuracy is likely to be very low ==> turn the prefetch off
	prefetch_offset = 0;
      }
      // new learning phase starts
      os_reset();
      return;
    }
  }
  INCREMENT(os.p,NOFFSETS); // prepare to test the next offset
}


//######################################################################################
//                               OFFSET PREFETCHER
//######################################################################################

// Issue at most one prefetch request. The prefetch line address is obtained by adding
// the prefetch offset to the current line address

int BestOffsetPrefetcher::issue_prefetch(t_addr lineaddr, int offset)
{
  if (offset == 0) {
    // The prefetcher is currently turned off.
    // Just push the line address into the delay queue for best-offset learning.
    dq_push(lineaddr);
    return 0;
  }
  if (! SAMEPAGE(lineaddr,lineaddr+offset)) {
    // crossing the page boundary, no prefetch request issued
    return 0;
  }
  if (get_l2_mshr_occupancy(0) < pt.mshr_threshold) {
    // prefetch into L2
    dq_push(lineaddr);
    return l2_prefetch_line(0,lineaddr<<LOGLINE,(lineaddr+offset)<<LOGLINE,FILL_L2);
  }
  // could not prefetch into L2
  // try to prefetch into LLC if prefetch accuracy not too bad
  if (pt.prefetch_score > LOW_SCORE) {
    return l2_prefetch_line(0,lineaddr<<LOGLINE,(lineaddr+offset)<<LOGLINE,FILL_LLC);
  }
  return 0;
}


//######################################################################################
//                               DPC2 INTERFACE
//######################################################################################


void BestOffsetPrefetcher::l2_prefetcher_initialize(int cpu_num)
{
  prefetch_offset = DEFAULT_OFFSET;
  rr_init();
  os_reset();
  dq_init();
  pt_init();
  int i,j;
  for (i=0; i<L2_SET_COUNT; i++) {
    for (j=0; j<L2_ASSOCIATIVITY; j++) {
      prefetch_bit[i][j] = 0;
    }
  }
}


void BestOffsetPrefetcher::l2_prefetcher_operate(int cpu_num, unsigned long long int addr, unsigned long long int ip, int cache_hit)
{
  t_addr lineaddr = addr >> LOGLINE;

  int s = l2_get_set(addr);
  int w = l2_get_way(0,addr,s);
  int l2_hit = (w>=0);
  int prefetched = 0;

  if (l2_hit) {
    // read the prefetch bit, and reset it
    prefetched = prefetch_bit[s][w];
    prefetch_bit[s][w] = 0;
  } else {
    pt_llc_access();
  }

  dq_pop();

  int prefetch_issued = 0;

  if (! l2_hit || prefetched) {
    os_learn_best_offset(lineaddr);
    prefetch_issued = issue_prefetch(lineaddr,prefetch_offset);
    if (prefetch_issued) {
      // assume the prefetch request is a L2 miss (we don't know actually)
      pt_llc_access();
    }
  }
}


void BestOffsetPrefetcher::l2_cache_fill(int cpu_num, unsigned long long int addr, int set, int way, int prefetch, unsigned long long int evicted_addr)
{
  // In this version of the DPC2 simulator, the "prefetch" boolean passed
  // as input here is not reset whenever a demand request hits in the L2
  // MSHR on an in-flight prefetch request. Fortunately, this is the information
  // we need for updating the RR table for best-offset learning.
  // However, the prefetch bit stored in the L2 is not completely accurate
  // (though hopefully this does not impact performance too much).
  // In a real hardware implementation of the BO prefetcher, we would distinguish
  // "prefetched" and "demand-requested", which are independent informations.

  t_addr lineaddr = addr >> LOGLINE;

  // write the prefetch bit
  int s = l2_get_set(addr);
  int w = l2_get_way(0,addr,s);
  prefetch_bit[s][w] = prefetch;

  // write the "right" bank of the RR table
  t_addr baselineaddr;
  if (prefetch || (prefetch_offset == 0)) {
    baselineaddr = lineaddr - prefetch_offset;
    if (SAMEPAGE(lineaddr,baselineaddr)) {
      rr_insert_right(baselineaddr);
    }
  }
}

void BestOffsetPrefetcher::calculatePrefetch(const PacketPtr &pkt,
                       std::vector<AddrPriority> &addresses)
{
    if (!pkt->req->hasPC()) {
        DPRINTF(HWPrefetch, "Ignoring request with no PC.\n");
        return;
    }

// Get required packet info
    Addr pkt_addr = pkt->getAddr();
    Addr pc = pkt->req->getPC();
    //bool is_secure = pkt->isSecure();
    //MasterID master_id = useMasterId ? pkt->req->masterId() : 0;
    //l2_prefetcher_operate(int cpu_num, unsigned long long int addr, unsigned long long int ip, int cache_hit);


}
#if 0
void l2_prefetcher_heartbeat_stats(int cpu_num)
{

}

void l2_prefetcher_warmup_stats(int cpu_num)
{

}

void l2_prefetcher_final_stats(int cpu_num)
{

}
#endif