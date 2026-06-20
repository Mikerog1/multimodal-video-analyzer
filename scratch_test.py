import os
import cv2

# Try to initialize video writer with 'avc1'
print("Python version:", os.sys.version)
print("OpenCV version:", cv2.__version__)
print("Current PATH:", os.environ.get('PATH', ''))

# Let's add core directory to PATH the traditional way
core_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'core'))
os.environ['PATH'] = core_dir + os.pathsep + os.environ.get('PATH', '')

# Test standard avc1 writer
fourcc = cv2.VideoWriter_fourcc(*'avc1')
test_path = 'test_avc1.mp4'
writer = cv2.VideoWriter(test_path, fourcc, 30.0, (640, 480))
is_opened = writer.isOpened()
print("avc1 writer opened (with traditional PATH modification):", is_opened)
if is_opened:
    writer.release()
    if os.path.exists(test_path):
        os.remove(test_path)

# Test with os.add_dll_directory
dll_cookie = None
if hasattr(os, 'add_dll_directory'):
    try:
        dll_cookie = os.add_dll_directory(core_dir)
        print("Successfully added DLL directory using os.add_dll_directory")
    except Exception as e:
        print("Failed to add DLL directory:", e)

fourcc2 = cv2.VideoWriter_fourcc(*'avc1')
test_path2 = 'test_avc1_dll.mp4'
writer2 = cv2.VideoWriter(test_path2, fourcc2, 30.0, (640, 480))
is_opened2 = writer2.isOpened()
print("avc1 writer opened (with os.add_dll_directory):", is_opened2)
if is_opened2:
    writer2.release()
    if os.path.exists(test_path2):
        os.remove(test_path2)

if dll_cookie:
    dll_cookie.close()
