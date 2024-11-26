# example.scenic

# Load X-Plane world model
from scenic.simulators.xplane.model import *

scenario = """
Plane at 5 @ 1000

param 'sim/weather/cloud_type[0]' = Uniform(0, 1, 2, 3, 4, 5)
setup_time = 10
simulation_length = 60
"""

