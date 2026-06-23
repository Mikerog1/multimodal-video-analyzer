import sys
import os

print("[+] Verifying Qwen2-VL package imports...")
try:
    import torch
    import transformers
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    print("[+] All packages (torch, transformers, qwen_vl_utils) are successfully imported!")
except ImportError as e:
    print(f"[-] ImportError: {e}")
    sys.exit(1)

print("[+] Verifying Qwen2-VL configuration loading...")
try:
    from transformers import Qwen2VLConfig
    config = Qwen2VLConfig.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    print(f"[+] Configuration for Qwen2-VL loaded successfully: {config.model_type}")
except Exception as e:
    print(f"[-] Config load failed: {e}")
    sys.exit(1)

print("[+] All automated tests for Qwen2-VL completed successfully!")
