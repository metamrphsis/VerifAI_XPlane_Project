# verifai_interface.py

from xpc import XPlaneConnect
import time

def main():
    xc = XPlaneConnect()
    try:
        while True:
            # Example: Get plane's position
            position = xc.getPOSI()
            print(f"Plane Position: {position}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interface terminated.")

if __name__ == "__main__":
    main()

# # verifai_interface.py
# 
# from xpc import XPlaneConnect
# import time
# 
# def main():
#     xc = XPlaneConnect()
#     try:
#         while True:
#             # Example: Get plane's position
#             position = xc.getPOSI()
#             print(f"Plane Position: {position}")
#             time.sleep(1)
#     except KeyboardInterrupt:
#         print("Interface terminated.")
# 
# if __name__ == "__main__":
#     main()

