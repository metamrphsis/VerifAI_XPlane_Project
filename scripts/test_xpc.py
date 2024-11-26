# test_xpc.py

from xpc import XPlaneConnect

xc = XPlaneConnect()
posi = xc.getPOSI()
print(f"Plane Position: {posi}")

