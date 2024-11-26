import socket
import struct
import logging

# Configure logging for debugging purposes
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class XPlaneConnect(object):
    """XPlaneConnect (XPC) facilitates communication to and from the XPCPlugin."""
    socket = None

    # Basic Functions
    def __init__(self, xpHost='localhost', xpPort=49007, port=0, timeout=100):
        """
        Sets up a new connection to an X-Plane Connect plugin running in X-Plane.

        Args:
            xpHost (str): The hostname of the machine running X-Plane.
            xpPort (int): The port on which the XPC plugin is listening. Usually 49007.
            port (int): The port which will be used to send and receive data. 0 lets the OS choose.
            timeout (int): The period (in milliseconds) after which read attempts will fail.
        """
        logger.debug(f"Initializing XPlaneConnect with host={xpHost}, xpPort={xpPort}, port={port}, timeout={timeout}")

        # Validate parameters
        try:
            xpIP = socket.gethostbyname(xpHost)
            logger.debug(f"Resolved {xpHost} to {xpIP}")
        except socket.gaierror:
            logger.error("Unable to resolve xpHost.")
            raise ValueError("Unable to resolve xpHost.")

        if not (0 <= xpPort <= 65535):
            logger.error("The specified X-Plane port is not a valid port number.")
            raise ValueError("The specified X-Plane port is not a valid port number.")
        if not (0 <= port <= 65535):
            logger.error("The specified port is not a valid port number.")
            raise ValueError("The specified port is not a valid port number.")
        if timeout < 0:
            logger.error("timeout must be non-negative.")
            raise ValueError("timeout must be non-negative.")

        # Setup XPlane IP and port
        self.xpDst = (xpIP, xpPort)
        logger.debug(f"Destination set to IP: {xpIP}, Port: {xpPort}")

        # Create and bind socket
        clientAddr = ("0.0.0.0", port)
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            self.socket.bind(clientAddr)
            logger.debug(f"Socket bound to {clientAddr}")
        except socket.error as e:
            logger.error(f"Failed to create or bind socket: {e}")
            raise

        # Set socket timeout
        timeout_seconds = timeout / 1000.0
        self.socket.settimeout(timeout_seconds)
        logger.debug(f"Socket timeout set to {timeout_seconds} seconds")

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
            logger.debug("Socket closed.")
            self.socket = None

    def sendUDP(self, buffer):
        """Sends a message over the underlying UDP socket."""
        # Preconditions
        if len(buffer) == 0:
            logger.error("sendUDP: buffer is empty.")
            raise ValueError("sendUDP: buffer is empty.")

        try:
            self.socket.sendto(buffer, self.xpDst)
            logger.debug(f"Sent UDP data to {self.xpDst}: {buffer}")
        except socket.error as e:
            logger.error(f"Failed to send UDP data: {e}")
            raise

    def readUDP(self):
        """Reads a message from the underlying UDP socket."""
        try:
            data, addr = self.socket.recvfrom(16384)
            logger.debug(f"Received UDP data from {addr}: {data}")
            return data
        except socket.timeout:
            logger.warning("UDP read timed out.")
            return None
        except socket.error as e:
            logger.error(f"Failed to read UDP data: {e}")
            raise

    # Configuration
    def setCONN(self, port):
        """
        Sets the port on which the client sends and receives data.

        Args:
            port (int): The new port to use.
        """
        logger.debug(f"Setting connection to port {port}")

        # Validate parameters
        if not (0 <= port <= 65535):
            logger.error("The specified port is not a valid port number.")
            raise ValueError("The specified port is not a valid port number.")

        # Send command
        try:
            buffer = struct.pack("<4sxH", b"CONN", port)
            self.sendUDP(buffer)
            logger.debug(f"Sent CONN command with port {port}")
        except struct.error as e:
            logger.error(f"Struct packing failed in setCONN: {e}")
            raise

        # Rebind socket
        clientAddr = ("0.0.0.0", port)
        timeout = self.socket.gettimeout()
        self.socket.close()
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            self.socket.bind(clientAddr)
            self.socket.settimeout(timeout)
            logger.debug(f"Rebound socket to {clientAddr} with timeout {timeout}")
        except socket.error as e:
            logger.error(f"Failed to rebind socket in setCONN: {e}")
            raise

        # Read response
        try:
            buffer = self.socket.recv(1024)
            logger.debug(f"Received response for CONN: {buffer}")
        except socket.timeout:
            logger.warning("CONN command response timed out.")
            raise TimeoutError("CONN command response timed out.")
        except socket.error as e:
            logger.error(f"Failed to receive CONN response: {e}")
            raise

    def pauseSim(self, pause):
        """
        Pauses or un-pauses the physics simulation engine in X-Plane.

        Args:
            pause (bool or int): True (1) to pause the simulation; False (0) to resume.
        """
        pause_value = int(bool(pause))
        if pause_value not in (0, 1):
            logger.error("Invalid argument for pause command.")
            raise ValueError("Invalid argument for pause command.")

        try:
            buffer = struct.pack("<4sxB", b"SIMU", pause_value)
            self.sendUDP(buffer)
            logger.debug(f"Sent SIMU command with pause={pause_value}")
        except struct.error as e:
            logger.error(f"Struct packing failed in pauseSim: {e}")
            raise

    # X-Plane UDP Data
    def readDATA(self):
        """
        Reads X-Plane data.

        Returns:
            list or None: A 2-dimensional list containing 0 or more rows of data.
                          Each sublist has 9 elements: row number and 8 data elements.
                          Returns None if no data is received.
        """
        buffer = self.readUDP()
        if buffer is None:
            logger.warning("No data received in readDATA.")
            return None
        if len(buffer) < 6:
            logger.warning(f"Received buffer too short in readDATA: {len(buffer)} bytes.")
            return None
        rows = (len(buffer) - 5) // 36
        data = []
        try:
            for i in range(rows):
                row_data = struct.unpack_from("9f", buffer, 5 + 36 * i)
                data.append(row_data)
                logger.debug(f"Read DATA row {i}: {row_data}")
            return data
        except struct.error as e:
            logger.error(f"Struct unpacking failed in readDATA: {e}")
            raise

    def sendDATA(self, data):
        """
        Sends X-Plane data over the underlying UDP socket.

        Args:
            data (list): An array of values representing data rows to be set.
                         Each sublist should have 9 elements:
                         [row_number (0-134), value1, value2, ..., value8]
        """
        if len(data) > 134:
            logger.error("Too many rows in data.")
            raise ValueError("Too many rows in data.")

        try:
            buffer = struct.pack("<4sx", b"DATA")
            for row in data:
                if len(row) != 9:
                    logger.error(f"Row does not contain exactly 9 values: {row}")
                    raise ValueError(f"Row does not contain exactly 9 values: {row}")
                buffer += struct.pack("<I8f", *row)
                logger.debug(f"Packed DATA row: {row}")
            self.sendUDP(buffer)
            logger.debug("Sent DATA command.")
        except struct.error as e:
            logger.error(f"Struct packing failed in sendDATA: {e}")
            raise

    # Position
    def getPOSI(self, ac=0):
        """
        Gets position information for the specified aircraft.

        Args:
            ac (int): The aircraft to get the position of. 0 is the main/player aircraft.

        Returns:
            tuple: (Latitude, Longitude, Altitude, Pitch, Roll, True Heading)
        """
        logger.debug(f"Requesting POSI for aircraft {ac}")

        try:
            # Send request
            buffer = struct.pack("<4sxB", b"GETP", ac)
            self.sendUDP(buffer)
            logger.debug(f"Sent GETP command for aircraft {ac}")

            # Read response
            resultBuf = self.readUDP()
            if resultBuf is None:
                logger.error("No response received for POSI request.")
                raise TimeoutError("No response received for POSI request.")

            if len(resultBuf) != 46:
                logger.error(f"Unexpected response length for POSI: {len(resultBuf)} bytes.")
                raise ValueError("Unexpected response length.")

            result = struct.unpack("<4sxBdddffff", resultBuf)
            if result[0] != b"POSI":
                logger.error(f"Unexpected header in POSI response: {result[0]}")
                raise ValueError("Unexpected header in POSI response.")

            # Extract position data
            posi_data = result[2:]
            logger.debug(f"Received POSI data for aircraft {ac}: {posi_data}")
            return posi_data
        except struct.error as e:
            logger.error(f"Struct unpacking failed in getPOSI: {e}")
            raise

    def sendPOSI(self, values, ac=0):
        """
        Sets position information on the specified aircraft.

        Args:
            values (list): A list containing up to 7 position values:
                           [Latitude (deg), Longitude (deg), Altitude (m above MSL),
                            Pitch (deg), Roll (deg), True Heading (deg), Gear (0=up, 1=down)]
                           Use -998 for any value you do not wish to change.
            ac (int): The aircraft to set the position of. 0 is the main/player aircraft.
        """
        logger.debug(f"Setting POSI for aircraft {ac} with values: {values}")

        # Preconditions
        if not (1 <= len(values) <= 7):
            logger.error("Must have between 1 and 7 items in values.")
            raise ValueError("Must have between 1 and 7 items in values.")
        if not (0 <= ac <= 20):
            logger.error("Aircraft number must be between 0 and 20.")
            raise ValueError("Aircraft number must be between 0 and 20.")

        try:
            # Pack message
            buffer = struct.pack("<4sxB", b"POSI", ac)
            for i in range(7):
                val = -998.0
                if i < len(values):
                    val = float(values[i])
                buffer += struct.pack("<f", val)
                logger.debug(f"Packed POSI value {i}: {val}")
            self.sendUDP(buffer)
            logger.debug(f"Sent POSI command for aircraft {ac}")
        except struct.error as e:
            logger.error(f"Struct packing failed in sendPOSI: {e}")
            raise

    # Controls
    def getCTRL(self, ac=0):
        """
        Gets the control surface information for the specified aircraft.

        Args:
            ac (int): The aircraft to get the control surfaces of. 0 is the main/player aircraft.

        Returns:
            tuple: Control surface states.
        """
        logger.debug(f"Requesting CTRL for aircraft {ac}")

        try:
            # Send request
            buffer = struct.pack("<4sxB", b"GETC", ac)
            self.sendUDP(buffer)
            logger.debug(f"Sent GETC command for aircraft {ac}")

            # Read response
            resultBuf = self.readUDP()
            if resultBuf is None:
                logger.error("No response received for CTRL request.")
                raise TimeoutError("No response received for CTRL request.")

            if len(resultBuf) != 31:
                logger.error(f"Unexpected response length for CTRL: {len(resultBuf)} bytes.")
                raise ValueError("Unexpected response length.")

            # Unpack response
            result = struct.unpack("<4sffffbfBf", resultBuf)
            if result[0] != b"CTRL":
                logger.error(f"Unexpected header in CTRL response: {result[0]}")
                raise ValueError("Unexpected header in CTRL response.")

            # Extract control data
            ctrl_data = result[1:7] + (result[8],)
            logger.debug(f"Received CTRL data for aircraft {ac}: {ctrl_data}")
            return ctrl_data
        except struct.error as e:
            logger.error(f"Struct unpacking failed in getCTRL: {e}")
            raise

    def sendCTRL(self, values, ac=0):
        """
        Sets control surface information on the specified aircraft.

        Args:
            values (list): A list containing up to 7 control surface values:
                           [Latitudinal Stick, Longitudinal Stick, Rudder Pedals,
                            Throttle, Gear, Flaps, Speedbrakes]
                           Use -998 for any value you do not wish to change.
            ac (int): The aircraft to set the control surfaces of. 0 is the main/player aircraft.
        """
        logger.debug(f"Setting CTRL for aircraft {ac} with values: {values}")

        # Preconditions
        if not (1 <= len(values) <= 7):
            logger.error("Must have between 1 and 7 items in values.")
            raise ValueError("Must have between 1 and 7 items in values.")
        if not (0 <= ac <= 20):
            logger.error("Aircraft number must be between 0 and 20.")
            raise ValueError("Aircraft number must be between 0 and 20.")

        try:
            # Pack message
            buffer = struct.pack("<4sx", b"CTRL")
            for i in range(6):
                val = -998.0
                if i < len(values):
                    val = float(values[i])
                if i == 4:
                    # Gear control as byte
                    if abs(val + 998) < 1e-4:
                        packed_val = -1  # Indicates no change
                    else:
                        packed_val = int(val)
                    buffer += struct.pack("b", packed_val)
                    logger.debug(f"Packed CTRL value {i} (Gear): {packed_val}")
                else:
                    buffer += struct.pack("<f", val)
                    logger.debug(f"Packed CTRL value {i}: {val}")

            # Aircraft ID
            buffer += struct.pack("<B", ac)
            logger.debug(f"Packed CTRL aircraft ID: {ac}")

            # Optional Speedbrakes
            if len(values) == 7:
                speedbrakes = float(values[6])
                buffer += struct.pack("<f", speedbrakes)
                logger.debug(f"Packed CTRL speedbrakes: {speedbrakes}")

            # Send
            self.sendUDP(buffer)
            logger.debug(f"Sent CTRL command for aircraft {ac}")
        except struct.error as e:
            logger.error(f"Struct packing failed in sendCTRL: {e}")
            raise

    # DREF Manipulation
    def sendDREF(self, dref, values):
        """
        Sets the specified dataref to the specified value.

        Args:
            dref (str): The name of the dataref to set.
            values (float or list): The value or list of values to set for the dataref.
        """
        logger.debug(f"Setting single DREF: {dref} with values: {values}")
        self.sendDREFs([dref], [values])

    def sendDREFs(self, drefs, values):
        """
        Sets the specified datarefs to the specified values.

        Args:
            drefs (list): A list of dataref names to set.
            values (list): A list of values or lists of values to set for each dataref.
        """
        logger.debug(f"Setting multiple DREFs: {drefs} with values: {values}")

        if len(drefs) != len(values):
            logger.error("drefs and values must have the same number of elements.")
            raise ValueError("drefs and values must have the same number of elements.")

        try:
            buffer = struct.pack("<4sx", b"DREF")
            for dref, value in zip(drefs, values):
                # Validate dref
                if not (0 < len(dref) <= 255):
                    logger.error("dref must be a non-empty string less than 256 characters.")
                    raise ValueError("dref must be a non-empty string less than 256 characters.")

                # Validate value
                if value is None:
                    logger.error("value must be a scalar or sequence of floats.")
                    raise ValueError("value must be a scalar or sequence of floats.")

                # Pack message
                if isinstance(value, (list, tuple)):
                    if len(value) > 255:
                        logger.error("value must have less than 256 items.")
                        raise ValueError("value must have less than 256 items.")
                    fmt = f"<B{len(dref)}sB{len(value)}f"
                    packed = struct.pack(fmt, len(dref), dref.encode(), len(value), *value)
                    logger.debug(f"Packed DREF {dref} with multiple values: {value}")
                else:
                    fmt = f"<B{len(dref)}sBf"
                    packed = struct.pack(fmt, len(dref), dref.encode(), 1, value)
                    logger.debug(f"Packed DREF {dref} with single value: {value}")
                buffer += packed

            # Send
            self.sendUDP(buffer)
            logger.debug("Sent DREF command.")
        except struct.error as e:
            logger.error(f"Struct packing failed in sendDREFs: {e}")
            raise

    def getDREF(self, dref):
        """
        Gets the value of an X-Plane dataref.

        Args:
            dref (str): The name of the dataref to get.

        Returns:
            list: The value(s) of the requested dataref.
        """
        logger.debug(f"Getting single DREF: {dref}")
        return self.getDREFs([dref])[0]

    def getDREFs(self, drefs):
        """
        Gets the value of one or more X-Plane datarefs.

        Args:
            drefs (list): The names of the datarefs to get.

        Returns:
            list: A list of tuples containing the values of the requested datarefs.
        """
        logger.debug(f"Getting multiple DREFs: {drefs}")

        try:
            # Send request
            buffer = struct.pack("<4sxB", b"GETD", len(drefs))
            for dref in drefs:
                fmt = f"<B{len(dref)}s"
                buffer += struct.pack(fmt, len(dref), dref.encode())
                logger.debug(f"Packed GETD for DREF: {dref}")
            self.sendUDP(buffer)

            # Read and parse response
            response = self.readUDP()
            if response is None:
                logger.error("No response received for GETD request.")
                raise TimeoutError("No response received for GETD request.")

            if len(response) < 6:
                logger.error(f"Response too short for GETD: {len(response)} bytes.")
                raise ValueError("Response too short for GETD.")

            resultCount = struct.unpack_from("B", response, 5)[0]
            logger.debug(f"Number of DREFs received: {resultCount}")
            offset = 6
            result = []
            for i in range(resultCount):
                if offset >= len(response):
                    logger.error("Incomplete response received for GETD.")
                    raise ValueError("Incomplete response received for GETD.")
                rowLen = struct.unpack_from("B", response, offset)[0]
                offset += 1
                if offset + rowLen * 4 > len(response):
                    logger.error("Insufficient data for DREF values.")
                    raise ValueError("Insufficient data for DREF values.")
                fmt = f"<{rowLen}f"
                row = struct.unpack_from(fmt, response, offset)
                result.append(row)
                logger.debug(f"Received DREF {i+1}: {row}")
                offset += rowLen * 4
            return result
        except struct.error as e:
            logger.error(f"Struct unpacking failed in getDREFs: {e}")
            raise

    # Drawing
    def sendTEXT(self, msg, x=-1, y=-1):
        """
        Sets a message that X-Plane will display on the screen.

        Args:
            msg (str): The string to display on the screen.
            x (int): The distance in pixels from the left edge of the screen to display the message.
                     A value of -1 indicates that the default horizontal position should be used.
            y (int): The distance in pixels from the bottom edge of the screen to display the message.
                     A value of -1 indicates that the default vertical position should be used.
        """
        logger.debug(f"Sending TEXT: '{msg}' at position x={x}, y={y}")

        if y < -1:
            logger.error("y must be greater than or equal to -1.")
            raise ValueError("y must be greater than or equal to -1.")

        if msg is None:
            msg = ""

        msgLen = len(msg)
        if msgLen > 255:
            logger.error("Message length exceeds 255 characters.")
            raise ValueError("Message length exceeds 255 characters.")

        try:
            # Pack message
            fmt = f"<4sxiiB{msgLen}s"
            buffer = struct.pack(fmt, b"TEXT", x, y, msgLen, msg.encode())
            self.sendUDP(buffer)
            logger.debug(f"Sent TEXT command with message: {msg}")
        except struct.error as e:
            logger.error(f"Struct packing failed in sendTEXT: {e}")
            raise

    def sendVIEW(self, view):
        """
        Sets the camera view in X-Plane.

        Args:
            view (int): The view to use. The ViewType class provides named constants for known views.
        """
        logger.debug(f"Sending VIEW command with view type: {view}")

        # Preconditions
        if not (ViewType.Forwards <= view <= ViewType.FullscreenNoHud):
            logger.error("Unknown view command.")
            raise ValueError("Unknown view command.")

        try:
            # Pack buffer
            buffer = struct.pack("<4sxi", b"VIEW", view)
            self.sendUDP(buffer)
            logger.debug(f"Sent VIEW command with view type: {view}")
        except struct.error as e:
            logger.error(f"Struct packing failed in sendVIEW: {e}")
            raise

    def sendWYPT(self, op, points):
        """
        Adds, removes, or clears waypoints.

        Waypoints are three-dimensional points on or above the Earth's surface that are
        represented visually in the simulator. Each point consists of a latitude and longitude
        expressed in fractional degrees and an altitude expressed as meters above sea level.

        Args:
            op (int): The operation to perform.
                      1 - Add waypoints,
                      2 - Remove waypoints,
                      3 - Clear all waypoints.
            points (list): A sequence of floating-point values representing latitude, longitude,
                           and altitude triples. The length of this array should always be divisible by 3.
        """
        logger.debug(f"Sending WYPT command with operation {op} and points: {points}")

        if op not in (1, 2, 3):
            logger.error("Invalid operation specified for WYPT.")
            raise ValueError("Invalid operation specified for WYPT.")
        if len(points) % 3 != 0:
            logger.error("Points should be divisible by 3 for WYPT.")
            raise ValueError("Points should be divisible by 3 for WYPT.")
        if (len(points) // 3) > 255:
            logger.error("Too many points. You can only send 255 points at a time.")
            raise ValueError("Too many points. You can only send 255 points at a time.")

        try:
            if op == 3:
                # Clear all waypoints
                buffer = struct.pack("<4sxBB", b"WYPT", op, 0)
                logger.debug("Prepared WYPT command to clear all waypoints.")
            else:
                # Add or Remove waypoints
                buffer = struct.pack(f"<4sxBB{len(points)}f", b"WYPT", op, len(points), *points)
                logger.debug(f"Prepared WYPT command to {'add' if op ==1 else 'remove'} waypoints.")
            self.sendUDP(buffer)
            logger.debug("Sent WYPT command.")
        except struct.error as e:
            logger.error(f"Struct packing failed in sendWYPT: {e}")
            raise

    def sendCOMM(self, comm):
        """
        Sends a COMM message to X-Plane.

        Args:
            comm (str): The communication message to send.
        """
        logger.debug(f"Sending COMM message: '{comm}'")

        if comm is None:
            logger.error("comm must be non-empty.")
            raise ValueError("comm must be non-empty.")
        if not (0 < len(comm) <= 255):
            logger.error("comm must be a non-empty string less than 256 characters.")
            raise ValueError("comm must be a non-empty string less than 256 characters.")

        try:
            # Pack message
            fmt = f"<4sxB{len(comm)}s"
            buffer = struct.pack(fmt, b"COMM", len(comm), comm.encode())
            self.sendUDP(buffer)
            logger.debug(f"Sent COMM message: {comm}")
        except struct.error as e:
            logger.error(f"Struct packing failed in sendCOMM: {e}")
            raise

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
