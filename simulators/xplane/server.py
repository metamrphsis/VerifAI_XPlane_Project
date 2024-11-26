# >>>>>>> xpc.py <<<<<<<
import socket
import struct
import os

import os

# Change to the desired directory
os.chdir('/Users/fdup/VerifAI/src/verifai/simulators/xplane')
print('os.getcwd()', os.getcwd())

class XPlaneConnect(object):
    """XPlaneConnect (XPC) facilitates communication to and from the XPCPlugin."""
    socket = None

    # Basic Functions
    def __init__(self, xpHost='localhost', xpPort=49009, port=0, timeout=100):
        """Sets up a new connection to an X-Plane Connect plugin running in X-Plane.

            Args:
              xpHost: The hostname of the machine running X-Plane.
              xpPort: The port on which the XPC plugin is listening. Usually 49007.
              port: The port which will be used to send and receive data.
              timeout: The period (in milliseconds) after which read attempts will fail.
        """

        # Validate parameters
        xpIP = None
        try:
            xpIP = socket.gethostbyname(xpHost)
        except:
            raise ValueError("Unable to resolve xpHost.")

        if xpPort < 0 or xpPort > 65535:
            raise ValueError("The specified X-Plane port is not a valid port number.")
        if port < 0 or port > 65535:
            raise ValueError("The specified port is not a valid port number.")
        if timeout < 0:
            raise ValueError("timeout must be non-negative.")

        # Setup XPlane IP and port
        self.xpDst = (xpIP, xpPort)

        # Create and bind socket
        clientAddr = ("0.0.0.0", port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.socket.bind(clientAddr)
        timeout /= 1000.0
        self.socket.settimeout(timeout)

    def __del__(self):
        self.close()

    # Define __enter__ and __exit__ to support the `with` construct.
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        """Closes the specified connection and releases resources associated with it."""
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def sendUDP(self, buffer):
        """Sends a message over the underlying UDP socket."""
        # Preconditions
        if(len(buffer) == 0):
            raise ValueError("sendUDP: buffer is empty.")

        self.socket.sendto(buffer, 0, self.xpDst)

    def readUDP(self):
        """Reads a message from the underlying UDP socket."""
        return self.socket.recv(16384)

    # Configuration
    def setCONN(self, port):
        """Sets the port on which the client sends and receives data.

            Args:
              port: The new port to use.
        """

        #Validate parameters
        if port < 0 or port > 65535:
            raise ValueError("The specified port is not a valid port number.")

        #Send command
        buffer = struct.pack(b"<4sxH", b"CONN", port)
        self.sendUDP(buffer)

        #Rebind socket
        clientAddr = ("0.0.0.0", port)
        timeout = self.socket.gettimeout()
        self.socket.close()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.socket.bind(clientAddr)
        self.socket.settimeout(timeout)

        #Read response
        buffer = self.socket.recv(1024)

    def pauseSim(self, pause):
        """Pauses or un-pauses the physics simulation engine in X-Plane.

            Args:
              pause: True to pause the simulation; False to resume.
        """
        pause = int(pause)
        if pause < 0 or pause > 2:
            raise ValueError("Invalid argument for pause command.")

        buffer = struct.pack(b"<4sxB", b"SIMU", pause)
        self.sendUDP(buffer)

    # X-Plane UDP Data
    def readDATA(self):
        """Reads X-Plane data.

            Returns: A 2 dimensional array containing 0 or more rows of data. Each array
              in the result will have 9 elements, the first of which is the row number which
              that array represents data for, and the rest of which are the data elements in
              that row.
        """
        buffer = self.readUDP()
        if len(buffer) < 6:
            return None
        rows = (len(buffer) - 5) / 36
        data = []
        for i in range(rows):
            data.append(struct.unpack_from(b"9f", buffer, 5 + 36*i))
        return data

    def sendDATA(self, data):
        """Sends X-Plane data over the underlying UDP socket.

            Args:
              data: An array of values representing data rows to be set. Each array in `data`
                should have 9 elements, the first of which is a row number in the range (0-134),
                and the rest of which are the values to set for that data row.
        """
        if len(data) > 134:
            raise ValueError("Too many rows in data.")

        buffer = struct.pack(b"<4sx", b"DATA")
        for row in data:
            if len(row) != 9:
                raise ValueError("Row does not contain exactly 9 values. <" + str(row) + ">")
            buffer += struct.pack(b"<I8f", *row)
        self.sendUDP(buffer)

    # Position
    def getPOSI(self, ac=0):
        """Gets position information for the specified aircraft.

        Args:
          ac: The aircraft to get the position of. 0 is the main/player aircraft.
        """
        # Send request
        buffer = struct.pack(b"<4sxB", b"GETP", ac)
        self.sendUDP(buffer)

        # Read response
        resultBuf = self.readUDP()
        if len(resultBuf) != 46:
            raise ValueError("Unexpected response length.")

        result = struct.unpack(b"<4sxBdddffff", resultBuf)
        if result[0] != b"POSI":
            raise ValueError("Unexpected header: " + result[0])

        # Drop the header & ac from the return value
        return result[2:]

    def sendPOSI(self, values, ac=0):
        """Sets position information on the specified aircraft.

            Args:
              values: The position values to set. `values` is a array containing up to
                7 elements. If less than 7 elements are specified or any elment is set to `-998`,
                those values will not be changed. The elements in `values` corespond to the
                following:
                  * Latitude (deg)
                  * Longitude (deg)
                  * Altitude (m above MSL)
                  * Pitch (deg)
                  * Roll (deg)
                  * True Heading (deg)
                  * Gear (0=up, 1=down)
              ac: The aircraft to set the position of. 0 is the main/player aircraft.
        """
        # Preconditions
        if len(values) < 1 or len(values) > 7:
            raise ValueError("Must have between 0 and 7 items in values.")
        if ac < 0 or ac > 20:
            raise ValueError("Aircraft number must be between 0 and 20.")

        # Pack message
        buffer = struct.pack(b"<4sxB", b"POSI", ac)
        for i in range(7):
            val = -998
            if i < len(values):
                val = values[i]
            buffer += struct.pack(b"<f", val)

        # Send
        self.sendUDP(buffer)

    # Controls
    def getCTRL(self, ac=0):
        """Gets the control surface information for the specified aircraft.

        Args:
          ac: The aircraft to get the control surfaces of. 0 is the main/player aircraft.
        """
        # Send request
        buffer = struct.pack(b"<4sxB", b"GETC", ac)
        self.sendUDP(buffer)

        # Read response
        resultBuf = self.readUDP()
        if len(resultBuf) != 31:
            raise ValueError("Unexpected response length.")

        result = struct.unpack(b"<4sxffffbfBf", resultBuf)
        if result[0] != b"CTRL":
            raise ValueError("Unexpected header: " + result[0])

        # Drop the header from the return value
        result =result[1:7] + result[8:]
        return result

    def sendCTRL(self, values, ac=0):
        """Sets control surface information on the specified aircraft.

            Args:
              values: The control surface values to set. `values` is a array containing up to
                6 elements. If less than 6 elements are specified or any elment is set to `-998`,
                those values will not be changed. The elements in `values` corespond to the
                following:
                  * Latitudinal Stick [-1,1]
                  * Longitudinal Stick [-1,1]
                  * Rudder Pedals [-1, 1]
                  * Throttle [-1, 1]
                  * Gear (0=up, 1=down)
                  * Flaps [0, 1]
                  * Speedbrakes [-0.5, 1.5]
              ac: The aircraft to set the control surfaces of. 0 is the main/player aircraft.
        """
        # Preconditions
        if len(values) < 1 or len(values) > 7:
            raise ValueError("Must have between 0 and 6 items in values.")
        if ac < 0 or ac > 20:
            raise ValueError("Aircraft number must be between 0 and 20.")

        # Pack message
        buffer = struct.pack(b"<4sx", b"CTRL")
        for i in range(6):
            val = -998
            if i < len(values):
                val = values[i]
            if i == 4:
                val = -1 if (abs(val + 998) < 1e-4) else val
                buffer += struct.pack(b"b", val)
            else:
                buffer += struct.pack(b"<f", val)

        buffer += struct.pack(b"B", ac)
        if len(values) == 7:
            buffer += struct.pack(b"<f", values[6])

        # Send
        self.sendUDP(buffer)

    # DREF Manipulation
    def sendDREF(self, dref, values):
        """Sets the specified dataref to the specified value.

            Args:
              dref: The name of the datarefs to set.
              values: Either a scalar value or a sequence of values.
        """
        self.sendDREFs([dref], [values])

    def sendDREFs(self, drefs, values):
        """Sets the specified datarefs to the specified values.

            Args:
              drefs: A list of names of the datarefs to set.
              values: A list of scalar or vector values to set.
        """
        if len(drefs) != len(values):
            raise ValueError("drefs and values must have the same number of elements.")

        buffer = struct.pack(b"<4sx", b"DREF")
        for i in range(len(drefs)):
            dref = drefs[i]
            value = values[i]

            # Preconditions
            if len(dref) == 0 or len(dref) > 255:
                raise ValueError("dref must be a non-empty string less than 256 characters.")

            if value is None:
                raise ValueError("value must be a scalar or sequence of floats.")

            # Pack message
            if hasattr(value, "__len__"):
                if len(value) > 255:
                    raise ValueError("value must have less than 256 items.")
                fmt = "<B{0:d}sB{1:d}f".format(len(dref), len(value))
                buffer += struct.pack(fmt.encode(), len(dref), dref.encode(), len(value), *value)
            else:
                fmt = "<B{0:d}sBf".format(len(dref))
                buffer += struct.pack(fmt.encode(), len(dref), dref.encode(), 1, value)

        # Send
        self.sendUDP(buffer)

    def getDREF(self, dref):
        """Gets the value of an X-Plane dataref.

            Args:
              dref: The name of the dataref to get.

            Returns: A sequence of data representing the values of the requested dataref.
        """
        return self.getDREFs([dref])[0]

    def getDREFs(self, drefs):
        """Gets the value of one or more X-Plane datarefs.

            Args:
              drefs: The names of the datarefs to get.

            Returns: A multidimensional sequence of data representing the values of the requested
             datarefs.
        """
        # Send request
        buffer = struct.pack(b"<4sxB", b"GETD", len(drefs))
        for dref in drefs:
            fmt = "<B{0:d}s".format(len(dref))
            buffer += struct.pack(fmt.encode(), len(dref), dref.encode())
        self.sendUDP(buffer)

        # Read and parse response
        buffer = self.readUDP()
        resultCount = struct.unpack_from(b"B", buffer, 5)[0]
        offset = 6
        result = []
        for i in range(resultCount):
            rowLen = struct.unpack_from(b"B", buffer, offset)[0]
            offset += 1
            fmt = "<{0:d}f".format(rowLen)
            row = struct.unpack_from(fmt.encode(), buffer, offset)
            result.append(row)
            offset += rowLen * 4
        return result

    # Drawing
    def sendTEXT(self, msg, x=-1, y=-1):
        """Sets a message that X-Plane will display on the screen.

            Args:
              msg: The string to display on the screen
              x: The distance in pixels from the left edge of the screen to display the
                 message. A value of -1 indicates that the default horizontal position should
                 be used.
              y: The distance in pixels from the bottom edge of the screen to display the
                 message. A value of -1 indicates that the default vertical position should be
                 used.
        """
        if y < -1:
            raise ValueError("y must be greater than or equal to -1.")

        if msg == None:
            msg = ""

        msgLen = len(msg)

        # TODO: Multiple byte conversions
        buffer = struct.pack(b"<4sxiiB" + (str(msgLen) + "s").encode(), b"TEXT", x, y, msgLen, msg.encode())
        self.sendUDP(buffer)

    def sendVIEW(self, view):
        """Sets the camera view in X-Plane

            Args:
              view: The view to use. The ViewType class provides named constants
                    for known views.
        """
        # Preconditions
        if view < ViewType.Forwards or view > ViewType.FullscreenNoHud:
            raise ValueError("Unknown view command.")

        # Pack buffer
        buffer = struct.pack(b"<4sxi", b"VIEW", view)

        # Send message
        self.sendUDP(buffer)

    def sendWYPT(self, op, points):
        """Adds, removes, or clears waypoints. Waypoints are three dimensional points on or
           above the Earth's surface that are represented visually in the simulator. Each
           point consists of a latitude and longitude expressed in fractional degrees and
           an altitude expressed as meters above sea level.

            Args:
              op: The operation to perform. Pass `1` to add waypoints,
                `2` to remove waypoints, and `3` to clear all waypoints.
              points: A sequence of floating point values representing latitude, longitude, and
                altitude triples. The length of this array should always be divisible by 3.
        """
        if op < 1 or op > 3:
            raise ValueError("Invalid operation specified.")
        if len(points) % 3 != 0:
            raise ValueError("Invalid points. Points should be divisible by 3.")
        if len(points) / 3 > 255:
            raise ValueError("Too many points. You can only send 255 points at a time.")

        if op == 3:
            buffer = struct.pack(b"<4sxBB", b"WYPT", 3, 0)
        else:
            buffer = struct.pack(("<4sxBB" + str(len(points)) + "f").encode(), b"WYPT", op, len(points), *points)
        self.sendUDP(buffer)

    def sendCOMM(self, comm):
        '''Sets the specified datarefs to the specified values.
            Args:
              drefs: A list of names of the datarefs to set.
              values: A list of scalar or vector values to set.
        '''
        if comm == None:
            raise ValueError("comm must be non-empty.")

        buffer = struct.pack(b"<4sx", b"COMM")
        if len(comm) == 0 or len(comm) > 255:
            raise ValueError("comm must be a non-empty string less than 256 characters.")

        # Pack message
        fmt = "<B{0:d}s".format(len(comm))
        buffer += struct.pack(fmt.encode(), len(comm), comm.encode())

        # Send
        self.sendUDP(buffer)



class ViewType(object):
    Forwards = 73
    Down = 74
    Left = 75
    Right = 76
    Back = 77
    Tower = 78
    Runway = 79
    Chase = 80
    Follow = 81
    FollowWithPanel = 82
    Spot = 83
    FullscreenWithHud = 84
    FullscreenNoHud = 85






# >>>>>>> images.py <<<<<<<
import os

import numpy as np

try:
    import mss
    import cv2
except ImportError as e:
    raise RuntimeError('recording images requires the mss and cv2 packages') from e

sct = mss.mss()

def grab_image(monitor, resizeTo=None):
    img = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)
    if resizeTo is not None:
        return cv2.resize(img, resizeTo)
    return img

def write_video(images, filename='out.avi', fps=10.0):
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError('failed to initialize VideoWriter')
    try:
        for image in images:
            writer.write(image)
    finally:
        writer.release()













# >>>>>>> server.py <<<<<<<
import functools, argparse, select
import struct, array
import time
import sys
import math
import socket

import numpy as np
import pandas as pd
import yaml
from dotmap import DotMap

# try:
#     # from xpc import XPlaneConnect
#     import XPlaneConnect
# except ImportError as e:
#     raise RuntimeError('the X-Plane interface requires XPlaneConnect') from e

import verifai, verifai.server, verifai.falsifier, verifai.samplers
import verifai.simulators.xplane.utils.controller as simple_controller
from verifai.simulators.xplane.utils.geometry import (euclidean_dist, quaternion_for,
    initial_bearing, cross_track_distance, compute_heading_error)

position_dref = ["sim/flightmodel/position/local_x",
                 "sim/flightmodel/position/local_y",
                 "sim/flightmodel/position/local_z"]

orientation_dref = ["sim/flightmodel/position/theta",
                    "sim/flightmodel/position/phi",
                    "sim/flightmodel/position/q"]

velocity_drefs = [      # velocity and acceleration drefs to null between runs
    "sim/flightmodel/position/local_vx",
    "sim/flightmodel/position/local_vy",
    "sim/flightmodel/position/local_vz",
    "sim/flightmodel/position/local_ax",
    "sim/flightmodel/position/local_ay",
    "sim/flightmodel/position/local_az",
    "sim/flightmodel/position/P",
    "sim/flightmodel/position/Q",
    "sim/flightmodel/position/R",
]

fuel_dref = "sim/flightmodel/weight/m_fuel"

class XPlaneServer(verifai.server.Server):
    def __init__(self, sampling_data, monitor, options):
        super().__init__(sampling_data, monitor, options)

        self.runway_heading = options.runway.heading
        self.rh_sin, self.rh_cos = math.sin(self.runway_heading), math.cos(self.runway_heading)
        self.start_lat, self.start_lon = options.runway.start_lat, options.runway.start_lon
        self.end_lat, self.end_lon = options.runway.end_lat, options.runway.end_lon
        self.desired_heading = initial_bearing(self.start_lat, self.start_lon,
                                               self.end_lat, self.end_lon)
        self.origin_x, self.origin_z = options.runway.origin_x, options.runway.origin_z

        path = pd.read_csv(options.runway.elevations)
        self.x, self.y, self.z = path['x'].values, path['y'].values, path['z'].values
        assert len(self.x) == len(self.y), "latitude and longitude values do not align"

        self.controller = options.get('controller')
        self.frametime = 1.0 / options.get('framerate', 10.0)
        self.timeout = options.get('timeout', 8000)
        self.predicates = options.get('predicates', {})
        self.verbosity = options.get('verbosity', 0)

        video = options.get('video')
        if video is not None:
            region = video['region']
            size = video.get('size')
            if size is not None:
                size = (size['width'], size['height'])
            # import utils.images as im
            # self.grab_image = lambda: im.grab_image(region, resizeTo=size)
            self.grab_image = lambda: grab_image(region, resizeTo=size)
        else:
            self.grab_image = None

        try:
            self.xpcserver = XPlaneConnect(timeout=self.timeout)
            input('Please start a new flight and then press ENTER.')
            self.initFuel = list(self.xpcserver.getDREF(fuel_dref))
            theta, phi, q = self.xpcserver.getDREFs(orientation_dref)
            self.initTheta, self.initPhi = math.radians(theta[0]), math.radians(phi[0])
            x, y, z = self.xpcserver.getDREFs(position_dref)
            self.initPosition = (x[0], y[0], z[0])
        except socket.timeout as e:
            raise RuntimeError('unable to connect to X-Plane') from e

    def evaluate_sample(self, sample):
        simulation_data = self.run_simulation(sample)
        value = 0 if self.monitor is None else self.monitor.evaluate(simulation_data)
        return value

    def run_simulation(self, sample):
        # Get runway endpoints
        start_lat, start_lon = self.start_lat, self.start_lon
        end_lat, end_lon = self.end_lat, self.end_lon

        # Extract test parameters from sample
        if isinstance(self.sampler, verifai.ScenicSampler):
            # special case for parameters whose names aren't valid identifiers
            # TODO generalize this mechanism to other sampler types?
            params = self.sampler.paramDictForSample(sample)
        else:
            params = sample.params._asdict()
        simulation_time = params.pop('simulation_length', 30)
        setup_time = params.pop('setup_time', 2)
        # anything left in params we assume to be a dref
        custom_drefs = params
        # get plane position and heading
        plane = sample.objects[0]
        heading = plane.heading
        print('plane.position = ', plane.position)
        # plane_x_sc, plane_z_sc = plane.position
        plane_x_sc, plane_y_sc, plane_z_sc = plane.position

        # Compute initial plane XZ position and heading in X-Plane local coordinates
        start_heading = self.runway_heading - heading
        plane_x = self.origin_x + (self.rh_cos * plane_x_sc) + (self.rh_sin * plane_z_sc)
        plane_z = self.origin_z - (self.rh_cos * plane_z_sc) + (self.rh_sin * plane_x_sc)

        # Find correct Y position for plane
        x_y = (plane_x, plane_z)
        euclid_dists = np.array([euclidean_dist(x_y, (self.x[i], self.z[i]))
                                for i in range(len(self.x))])
        best_elev = np.argmin(euclid_dists)
        plane_y = self.y[best_elev]

        # Reset plane and apply brake
        self.xpcserver.sendDREFs(velocity_drefs, [0]*len(velocity_drefs))
        self.xpcserver.sendCTRL([0, 0, 0, 0])
        self.xpcserver.sendCOMM("sim/operation/fix_all_systems")
        self.xpcserver.sendDREF(fuel_dref, self.initFuel)
        self.xpcserver.sendDREF("sim/flightmodel/controls/parkbrake", 1)

        # Set new plane position
        self.xpcserver.sendDREFs(position_dref, [plane_x, plane_y, plane_z])
        # Set new plane orientation
        quaternion = quaternion_for(self.initTheta, self.initPhi, start_heading)
        self.xpcserver.sendDREF("sim/flightmodel/position/q", quaternion)
        # Set any other specified drefs
        self.xpcserver.sendDREFs(list(custom_drefs.keys()), list(custom_drefs.values()))

        # Wait for weather, etc. to stabilize, then release brake
        time.sleep(setup_time)
        self.xpcserver.sendCOMM("sim/operation/fix_all_systems")
        time.sleep(0.1)
        self.xpcserver.sendDREF("sim/flightmodel/controls/parkbrake", 0)

        # Execute a run
        if self.verbosity >= 1:
            print('Starting run...')
        start = time.time()
        current = start
        lats, lons, psis, ctes, hes, times, images = [], [], [], [], [], [], []
        while current - start < simulation_time:
            times.append(current - start)
            # Get current plane state
            # Use modified getPOSI to get lat/lon in double precision
            lat, lon, _, _, _, psi, _ = self.xpcserver.getPOSI()
            lats.append(lat); lons.append(lon); psis.append(psi)
            # Compute cross-track and heading errors
            cte = cross_track_distance(start_lat, start_lon, end_lat, end_lon, lat, lon)
            heading_err = compute_heading_error(self.desired_heading, psi)
            ctes.append(cte); hes.append(heading_err)
            # Run controller for one step, if desired
            if self.controller is not None:
                self.controller(self.xpcserver, lat, lon, psi, cte, heading_err)
            # Save screenshot for videos
            if self.grab_image is not None:
                images.append(self.grab_image())
            if self.verbosity >= 2:
                print(f'cte: {cte}; heading_err: {heading_err}')

            # Limit framerate (for fast controllers)
            wait_time = self.frametime - (time.time() - current)
            if wait_time > 0:
                time.sleep(wait_time)

            current = time.time()
        self.images = images

        # Do some simple checks to see if the plane has gotten stuck
        thresh = 0.000001
        end_point_check, mid_point_check = True, True
        if abs(lats[0] - lats[-1]) < thresh and abs(lons[0] - lons[-1]) < thresh:
            end_point_check = False
        num_lats = len(lats)
        if abs(lats[0] - lats[num_lats//2]) < thresh and abs(lons[0] - lons[num_lats//2]) < thresh:
            mid_point_check = False
        if not (mid_point_check or end_point_check):
            raise RuntimeError('Plane appears to have gotten stuck!')

        # Compute time series for each atomic predicate
        simulation_data = {}
        for name, predicate in self.predicates.items():
            series = [
                (time, predicate(lat, lon, psi, cte, he))
                for time, lat, lon, psi, cte, he in zip(times, lats, lons, psis, ctes, hes)
            ]
            simulation_data[name] = series
        return simulation_data


class XPlaneFalsifier(verifai.falsifier.mtl_falsifier):
    def __init__(self, monitor, sampler_type=None, sampler=None, sample_space=None,
                 falsifier_params={}, server_options={}):
        super().__init__(monitor, sampler_type=sampler_type, sampler=sampler,
                         sample_space=sample_space,
                         falsifier_params=falsifier_params, server_options=server_options)
        self.verbosity = falsifier_params.get('verbosity', 0)
        video = falsifier_params.get('video')
        if video is not None:
            # import utils.images       # will throw exception if required libraries not available
            self.video_threshold = video['threshold']
            self.video_framerate = video['framerate']
        else:
            self.video_threshold = None

    # def init_server(self, server_options):
    def init_server(self, server_options, server_class):
        samplingConfig = DotMap(sampler=self.sampler,
                                sampler_type=self.sampler_type,
                                sample_space=self.sample_space)
        self.server = XPlaneServer(samplingConfig, self.monitor, server_options)

    def populate_error_table(self, sample, rho, error=True):
        super().populate_error_table(sample, rho, error)
        if self.video_threshold is not None and rho <= self.video_threshold:
            if error:
                index = len(self.error_table.table) - 1
                name = f'error-{index}'
            else:
                index = len(self.safe_table.table) - 1
                name = f'safe-{index}'
            # import utils.images
            # utils.images.write_video(self.server.images, filename=name+'.avi',
            #                          fps=self.video_framerate)
            write_video(self.server.images, filename=name+'.avi',
                                     fps=self.video_framerate)


def run_test(configuration, runway, verbosity=0):
    # Load Scenic scenario
    print('Loading scenario...')
    sampler = verifai.ScenicSampler.fromScenario(configuration['scenario'])

    # Define predicates and specifications
    def nearcenterline(lat, lon, psi, cte, he):
        """MTL predicate 'cte < 1.5', i.e. within 1.5 m of centerline."""
        return 1.5 - abs(cte)
    predicates = { 'nearcenterline': nearcenterline }
    specification = [configuration.get('specification', 'G nearcenterline')]

    # Set up controller
    framerate = configuration['framerate']
    controller = simple_controller.control if configuration['controller'] else None

    # Get options for video recording
    video = configuration.get('video')
    if video is None or not video['record']:
        video = None
    else:
        video['framerate'] = framerate

    # Create falsifier (and underlying server)
    serverOptions = DotMap(port=8080, controller=controller, framerate=framerate,
                           predicates=predicates, verbosity=verbosity,
                           runway=runway, video=video)
    falsifierOptions = DotMap(
        n_iters=configuration['runs'],
        verbosity=verbosity, video=video,
        save_error_table=True, save_safe_table=True,
        error_table_path=configuration['error_table'],
        safe_table_path=configuration['safe_table'],
    )
    falsifier = XPlaneFalsifier(specification, sampler=sampler,
                                falsifier_params=falsifierOptions,
                                server_options=serverOptions)

    # Perform falsification
    try:
        falsifier.run_falsifier()
    except socket.timeout as e:
        raise RuntimeError('lost connection to X-Plane') from e
    finally:
        # Print error and safe tables
        if verbosity >= 1:
            print('Error Table (also saved to {}):'.format(falsifier.error_table_path))
            print(falsifier.error_table.table)
            print('Safe Table (also saved to {}):'.format(falsifier.safe_table_path))
            print(falsifier.safe_table.table)
        # Write out final cross-entropy distributions
        es = sampler.scenario.externalSampler
        if es is not None:
            es = es.sampler.domainSampler
            if isinstance(es, verifai.samplers.cross_entropy.CrossEntropySampler):
                if verbosity >= 1:
                    print('Cross-entropy distributions:')
                with open(configuration['cross_entropy'], 'w') as outfile:
                    for s in es.split_sampler.samplers:
                        if verbosity >= 1:
                            print(s.dist)
                        outfile.write(str(s.dist)+'\n')


def load_yaml(filename):
    print('filename', filename)
    with open(filename, 'r') as stream:
        print(stream)
        options = yaml.safe_load(stream)
    return options


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='experiment configuration file', default='config.yaml')
    parser.add_argument('-r', '--runway', help='runway configuration file', default='runway.yaml')
    parser.add_argument('-v', '--verbosity', type=int, default=0)
    args = parser.parse_args()

    # Parse runway configuration
    print('os.getcwd()', os.getcwd())
    print(args.runway)
    runway = load_yaml(args.runway)
    print(args.runway)
    # runway = load_yaml('runway.yaml')
    rads = runway['radians']
    runway_heading = runway['heading']
    if not rads:
        runway_heading = math.radians(runway_heading)
    runway_data = DotMap(
        heading=runway_heading, elevations=runway['elevations'],
        origin_x=runway['origin_X'], origin_z=runway['origin_Z'],
        start_lat=runway['start_lat'], start_lon=runway['start_lon'],
        end_lat=runway['end_lat'], end_lon=runway['end_lon']
    )

    # Parse experiment configuration
    configuration = load_yaml(args.config)

    run_test(configuration, runway_data, verbosity=args.verbosity)
