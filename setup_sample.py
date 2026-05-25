import os
import sys
import httpx

def main():
    target_dir = "input"
    os.makedirs(target_dir, exist_ok=True)
    
    target_file = os.path.join(target_dir, "sample.mp4")
    url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4"
    
    print("[+] Preparing sample video for detection test...")
    if os.path.exists(target_file):
        print(f"[+] Sample video already exists at {target_file}")
        return
        
    print(f"[+] Downloading sample video from: {url}")
    try:
        # Download using httpx which is already installed in the environment
        with httpx.Client(follow_redirects=True, timeout=30.0) as client:
            response = client.get(url)
            response.raise_for_status()
            
            with open(target_file, "wb") as f:
                f.write(response.content)
                
        print(f"[+] Download successful! Saved to: {target_file}")
        print(f"[+] File size: {os.path.getsize(target_file) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"[-] Failed to download sample video: {e}")
        print("[-] Please copy any video file (e.g., mp4) to the 'input/' folder manually and rename it to 'sample.mp4'.")
        sys.exit(1)

if __name__ == "__main__":
    main()
