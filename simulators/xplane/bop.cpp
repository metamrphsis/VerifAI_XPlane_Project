/* BOP [https://hal.inria.fr/hal-01254863/document] */

#include <bits/stdc++.h>

using namespace std;


template <class T> class InfiniteCache {
  public:
    class Entry {
      public:
        uint64_t key;
        uint64_t index;
        uint64_t tag;
        bool valid;
        T data;
    };

    Entry *erase(uint64_t key) {
        Entry *entry = this->find(key);
        if (!entry)
            return nullptr;
        entry->valid = false;
        this->last_erased_entry = *entry;
        int num_erased = this->entries.erase(key);
        assert(num_erased == 1);
        return &this->last_erased_entry;
    }

    /**
     * @return The old state of the entry that was written to.
     */
    Entry insert(uint64_t key, const T &data) {
        Entry *entry = this->find(key);
        if (entry != nullptr) {
            Entry old_entry = *entry;
            entry->data = data;
            return old_entry;
        }
        entries[key] = {key, 0, key, true, data};
        return {};
    }

    Entry *find(uint64_t key) {
        auto it = this->entries.find(key);
        if (it == this->entries.end())
            return nullptr;
        Entry &entry = (*it).second;
        assert(entry.tag == key && entry.valid);
        return &entry;
    }

  protected:
    Entry last_erased_entry;
    unordered_map<uint64_t, Entry> entries;
    bool debug = false;
};

template <class T> class SetAssociativeCache {
  public:
    class Entry {
      public:
        uint64_t key;
        uint64_t index;
        uint64_t tag;
        bool valid;
        T data;
    };

    SetAssociativeCache(int size, int num_ways)
        : size(size), num_ways(num_ways), num_sets(size / num_ways), entries(num_sets, vector<Entry>(num_ways)),
          cams(num_sets) {
        assert(size % num_ways == 0);
        for (int i = 0; i < num_sets; i += 1)
            for (int j = 0; j < num_ways; j += 1)
                entries[i][j].valid = false;
    }

    Entry *erase(uint64_t key) {
        Entry *entry = this->find(key);
        uint64_t index = key % this->num_sets;
        uint64_t tag = key / this->num_sets;
        auto &cam = cams[index];
        int num_erased = cam.erase(tag);
        if (entry)
            entry->valid = false;
        assert(entry ? num_erased == 1 : num_erased == 0);
        return entry;
    }

    /**
     * @return The old state of the entry that was written to.
     */
    Entry insert(uint64_t key, const T &data) {
        Entry *entry = this->find(key);
        if (entry != nullptr) {
            Entry old_entry = *entry;
            entry->data = data;
            return old_entry;
        }
        uint64_t index = key % this->num_sets;
        uint64_t tag = key / this->num_sets;
        vector<Entry> &set = this->entries[index];
        int victim_way = -1;
        for (int i = 0; i < this->num_ways; i += 1)
            if (!set[i].valid) {
                victim_way = i;
                break;
            }
        if (victim_way == -1) {
            victim_way = this->select_victim(index);
        }
        Entry &victim = set[victim_way];
        Entry old_entry = victim;
        victim = {key, index, tag, true, data};
        auto &cam = cams[index];
        if (old_entry.valid) {
            int num_erased = cam.erase(old_entry.tag);
            assert(num_erased == 1);
        }
        cam[tag] = victim_way;
        return old_entry;
    }

    Entry *find(uint64_t key) {
        uint64_t index = key % this->num_sets;
        uint64_t tag = key / this->num_sets;
        auto &cam = cams[index];
        if (cam.find(tag) == cam.end())
            return nullptr;
        int way = cam[tag];
        Entry &entry = this->entries[index][way];
        assert(entry.tag == tag && entry.valid);
        return &entry;
    }

  protected:
    /**
     * @return The way of the selected victim.
     */
    virtual int select_victim(uint64_t index) {
        /* random eviction policy if not overriden */
        return rand() % this->num_ways;
    }

    vector<Entry> get_valid_entries() {
        vector<Entry> valid_entries;
        for (int i = 0; i < num_sets; i += 1)
            for (int j = 0; j < num_ways; j += 1)
                if (entries[i][j].valid)
                    valid_entries.push_back(entries[i][j]);
        return valid_entries;
    }

    int size;
    int num_ways;
    int num_sets;
    vector<vector<Entry>> entries;
    vector<unordered_map<uint64_t, int>> cams;
    bool debug = false;
};

template <class T> class LRUSetAssociativeCache : public SetAssociativeCache<T> {
    typedef SetAssociativeCache<T> Super;

  public:
    LRUSetAssociativeCache(int size, int num_ways)
        : Super(size, num_ways), lru(this->num_sets, vector<uint64_t>(num_ways)) {}

    void set_mru(uint64_t key) { *this->get_lru(key) = this->t++; }

    void set_lru(uint64_t key) { *this->get_lru(key) = 0; }

  protected:
    /* @override */
    int select_victim(uint64_t index) {
        vector<uint64_t> &lru_set = this->lru[index];
        return min_element(lru_set.begin(), lru_set.end()) - lru_set.begin();
    }

    uint64_t *get_lru(uint64_t key) {
        uint64_t index = key % this->num_sets;
        uint64_t tag = key / this->num_sets;
        int way = this->cams[index][tag];
        return &this->lru[index][way];
    }

    vector<vector<uint64_t>> lru;
    uint64_t t = 0;
};

template <class T> class NMRUSetAssociativeCache : public SetAssociativeCache<T> {
    typedef SetAssociativeCache<T> Super;

  public:
    NMRUSetAssociativeCache(int size, int num_ways) : Super(size, num_ways), mru(this->num_sets) {}

    void set_mru(uint64_t key) {
        uint64_t index = key % this->num_sets;
        uint64_t tag = key / this->num_sets;
        int way = this->cams[index][tag];
        this->mru[index] = way;
    }

  protected:
    /* @override */
    int select_victim(uint64_t index) {
        int way = rand() % (this->num_ways - 1);
        if (way >= mru[index])
            way += 1;
        return way;
    }

    vector<int> mru;
};

template <class T> class LRUFullyAssociativeCache : public LRUSetAssociativeCache<T> {
    typedef LRUSetAssociativeCache<T> Super;

  public:
    LRUFullyAssociativeCache(int size) : Super(size, size) {}
};

template <class T> class NMRUFullyAssociativeCache : public NMRUSetAssociativeCache<T> {
    typedef NMRUSetAssociativeCache<T> Super;

  public:
    NMRUFullyAssociativeCache(int size) : Super(size, size) {}
};

template <class T> class DirectMappedCache : public SetAssociativeCache<T> {
    typedef SetAssociativeCache<T> Super;

  public:
    DirectMappedCache(int size) : Super(size, 1) {}
};

/** End Of Cache Framework **/


// RECENT REQUEST TABLE
class RecentRequestsTableData {
  public:
    uint64_t base_address;
};

class RecentRequestsTable : public DirectMappedCache<RecentRequestsTableData> {
    typedef DirectMappedCache<RecentRequestsTableData> Super;

  public:
    RecentRequestsTable(int size) : Super(size) {
        assert(__builtin_popcount(size) == 1);
        this->hash_w = __builtin_ctz(size);
    }

    Entry insert(uint64_t base_address) {
        uint64_t key = this->hash(base_address);
        return Super::insert(key, {base_address});
    }

    bool find(uint64_t base_address) {
        uint64_t key = this->hash(base_address);
        return (Super::find(key) != nullptr);
    }

    string log() {
        vector<string> headers({"Hash", "Base Address"});
        return Super::log(headers, this->write_data);
    }

  private:
    static void write_data(Entry &entry, Table &table, int row) {
        table.set_cell(row, 0, bitset<20>(entry.key).to_string());
        table.set_cell(row, 1, entry.data.base_address);
    }

    /* The RR table is accessed through a simple hash function. For instance, for a 256-entry RR table, we XOR the 8
     * least significant line address bits with the next 8 bits to obtain the table index. For 12-bit tags, we skip the
     * 8 least significant line address bits and extract the next 12 bits. */
    uint64_t hash(uint64_t input) {
        int next_w_bits = ((1 << hash_w) - 1) & (input >> hash_w);
        uint64_t output = ((1 << 20) - 1) & (next_w_bits ^ input);
        if (this->debug) {
            cerr << "[RR] hash( " << bitset<32>(input).to_string() << " ) = " << bitset<20>(output).to_string() << endl;
        }
        return output;
    }

    int hash_w;
};



// BEST OFFSET LEARNING

class BestOffsetLearning {
  public:
    BestOffsetLearning(int blocks_in_page) : blocks_in_page(blocks_in_page) {
        /* Useful offset values depend on the memory page size, as the BO prefetcher does not prefetch across page
         * boundaries. For instance, assuming 4KB pages and 64B lines, a page contains 64 lines, and there is no point
         * in considering offset values greater than 63. However, it may be useful to consider offsets greater than 63
         * for systems having superpages. */
        /* We propose a method for offset sampling that is algorithmic and not totally arbitrary: we include in our list
         * all the offsets between 1 and 256 whose prime factorization does not contain primes greater than 5. */
        /* Nothing prevents a BO prefetcher to use negative offset values. Although some applications might benefit from
         * negative offsets, we did not observe any benefit in our experiments. Hence we consider only positive offsets
         * in this study. */
        for (int i = 1; i < blocks_in_page; i += 1) {
            int n = i;
            for (int j = 2; j <= 5; j += 1)
                while (n % j == 0)
                    n /= j;
            if (n == 1)
                offset_list.push_back({i, 0});
        }
    }

    /**
     * @return The current best offset.
     */
    int test_offset(uint64_t block_number, RecentRequestsTable &recent_requests_table) {
        int page_offset = block_number % this->blocks_in_page;
        Entry &entry = this->offset_list[this->index_to_test];
        bool found =
            is_inside_page(page_offset - entry.offset) && recent_requests_table.find(block_number - entry.offset);
        if (this->debug) {
            cerr << "[BOL] testing offset=" << entry.offset << " with score=" << entry.score << endl;
            cerr << "[BOL] match=" << found << endl;
        }
        if (found) {
            entry.score += 1;
            if (entry.score > this->best_score) {
                this->best_score = entry.score;
                this->local_best_offset = entry.offset;
            }
        }
        this->index_to_test = (this->index_to_test + 1) % this->offset_list.size();
        /* test round termination */
        if (this->index_to_test == 0) {
            if (this->debug) {
                cerr << "[BOL] round=" << this->round << " finished" << endl;
            }
            this->round += 1;
            /* The current learning phase finishes at the end of a round when either of the two following events happens
             * first: one of the scores equals SCOREMAX, or the number of rounds equals ROUNDMAX (a fixed parameter). */
            if (this->best_score >= SCORE_MAX || this->round == ROUND_MAX) {
                if (this->best_score <= BAD_SCORE)
                    this->global_best_offset = 0; /* turn off prefetching */
                else
                    this->global_best_offset = this->local_best_offset;
                if (this->debug) {
                    cerr << "[BOL] learning phase finished, winner=" << this->global_best_offset << endl;
                    cerr << this->log();
                }
                /* reset all internal state */
                for (auto &entry : this->offset_list)
                    entry.score = 0;
                this->local_best_offset = 0;
                this->best_score = 0;
                this->round = 0;
            }
        }
        return this->global_best_offset;
    }

    string log() {
        Table table(2, offset_list.size() + 1);
        table.set_row(0, {"Offset", "Score"});
        for (unsigned i = 0; i < offset_list.size(); i += 1) {
            table.set_cell(i + 1, 0, offset_list[i].offset);
            table.set_cell(i + 1, 1, offset_list[i].score);
        }
        return table.to_string();
    }

    void set_debug_mode(bool enable) { this->debug = enable; }

  private:
    bool is_inside_page(int page_offset) { return (0 <= page_offset && page_offset < this->blocks_in_page); }

    class Entry {
      public:
        int offset;
        int score;
    };

    int blocks_in_page;
    vector<Entry> offset_list;

    int round = 0;
    int best_score = 0;
    int index_to_test = 0;
    int local_best_offset = 0;
    int global_best_offset = 1;

    const int SCORE_MAX = 31;
    const int ROUND_MAX = 100;
    const int BAD_SCORE = 1;

    bool debug = false;
};


// BEST OFFSET PREFETCHER
class BOP {
  public:
    BOP(int blocks_in_page = 64, int recent_requests_table_size = 256)
        : blocks_in_page(blocks_in_page), best_offset_learning(blocks_in_page),
          recent_requests_table(recent_requests_table_size) {}

    /**
     * @return A vector of block numbers that should be prefetched.
     */
    vector<uint64_t> access(uint64_t block_number) {
        uint64_t page_number = block_number / this->blocks_in_page;
        int page_offset = block_number % this->blocks_in_page;
        /* ... and if X and X + D lie in the same memory page, a prefetch request for line X + D is sent to the L3
         * cache. */
        if (this->debug) {
            cerr << "[BOP] block_number=" << block_number << endl;
            cerr << "[BOP] page_number=" << page_number << endl;
            cerr << "[BOP] page_offset=" << page_offset << endl;
            cerr << "[BOP] best_offset=" << this->prefetch_offset << endl;
        }
        /* The BO prefetcher is a degree-one prefetcher: it issues at most one prefetch per access. */
        vector<uint64_t> pred;
        if (is_inside_page(page_offset + this->prefetch_offset))
            pred.push_back(block_number + this->prefetch_offset);
        else if (this->debug)
            cerr << "[BOP] X and X + D do not lie in the same memory page, no prefetch issued" << endl;
        this->recent_requests_table.insert(block_number);
        int old_offset = this->prefetch_offset;
        /* On every eligible L2 read access (miss or prefetched hit), we test an offset di from the list. */
        this->prefetch_offset = this->best_offset_learning.test_offset(block_number, recent_requests_table);
        if (this->debug) {
            if (old_offset != this->prefetch_offset)
                cerr << "[BOP] offset changed from " << old_offset << " to " << this->prefetch_offset << endl;
            cerr << this->recent_requests_table.log();
            cerr << this->best_offset_learning.log();
        }
        return pred;
    }

  private:
    bool is_inside_page(int page_offset) { return (0 <= page_offset && page_offset < this->blocks_in_page); }

    int blocks_in_page;
    int prefetch_offset = 1; /* initialize best offset to 1? */

    BestOffsetLearning best_offset_learning;
    RecentRequestsTable recent_requests_table;

    bool debug = false;
};

class CacheData {
  public:
    bool prefetch;
};

class Cache : public LRUSetAssociativeCache<CacheData> {
    typedef LRUSetAssociativeCache<CacheData> Super;

  public:
    Cache(int size, int block_size, int num_ways, int cores, int page_size, string name)
        : Super(size / block_size, num_ways), bop_modules(cores, BOP(page_size / block_size)), name(name) {
        assert(size % block_size == 0);
        assert(page_size % block_size == 0);
        assert(__builtin_popcount(this->num_sets) == 1);
        assert(__builtin_popcount(this->num_ways) == 1);
    }

    void access(uint64_t address, string type, int core) {
        if (this->debug) {
            cerr << "[Cache] address=" << address << endl;
            cerr << "[Cache] type=" << type << endl;
            cerr << "[Cache] core=" << core << endl;
        }
        this->stats["Accesses"] += 1;
        uint64_t block_number = address / block_size;
        Entry *entry = this->find(block_number);
        bool trigger_access = false;
        if (entry) {
            if (this->debug) {
                cerr << "[Cache] hit" << endl;
            }
            this->set_mru(block_number);
            if (entry->data.prefetch) {
                this->stats["Prefetch Hits"] += 1;
                if (this->debug) {
                    cerr << "[Cache] prefetch hit" << endl;
                }
                entry->data.prefetch = false;
                trigger_access = true;
            }
        } else {
            this->stats["Misses"] += 1;
            if (this->debug) {
                cerr << "[Cache] miss" << endl;
            }
            this->fetch(block_number, false);
            trigger_access = true;
        }
        /* When a read request for line X accesses the L2 cache, if this is a miss or a prefetched hit ... */
        if (type != "Read Request" || !trigger_access)
            return;
        vector<uint64_t> to_prefetch = bop_modules[core].access(block_number);
        if (this->debug) {
            cerr << "[Cache] to_prefetch=<";
            for (unsigned i = 0; i < to_prefetch.size(); i += 1) {
                if (i != 0)
                    cerr << ", ";
                cerr << to_prefetch[i];
            }
            cerr << ">" << endl;
        }
        for (auto &block_number : to_prefetch)
            this->fetch(block_number, true);
    }

    vector<string> get_log_headers() {
        vector<string> log_headers = {"Name"};
        for (auto &x : this->stats)
            log_headers.push_back(x.first);
        return log_headers;
    }

    /**
     * Write log info in specified row of given table.
     */
    void log(Table &table, int row) {
        int col = 0;
        table.set_cell(row, col++, this->name);
        for (auto &x : this->stats)
            table.set_cell(row, col++, x.second);
    }

    uint64_t get_accesses() { return this->stats["Accesses"]; }

    void set_debug_mode(bool enable) {
        for (auto &module : bop_modules)
            module.set_debug_mode(enable);
        this->debug = enable;
    }

  private:
    void fetch(uint64_t block_number, bool prefetch) {
        if (this->find(block_number))
            return;
        if (prefetch)
            this->stats["Prefetches"] += 1;
        Entry old_entry = this->insert(block_number, {prefetch});
        this->set_mru(block_number);
        if (old_entry.valid && old_entry.data.prefetch)
            this->stats["Non-useful Prefetches"] += 1;
    }

    int block_size = 64;
    vector<BOP> bop_modules;
    string name;
    bool debug = false;
    unordered_map<string, uint64_t> stats = {
        {"Accesses", 0}, {"Misses", 0}, {"Prefetches", 0}, {"Prefetch Hits", 0}, {"Non-useful Prefetches", 0}};
};
