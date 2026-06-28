import base64
import requests
import json
import cv2

class VLMClient:
    def __init__(self, backend_type: str = "gemini", api_key: str = None, api_url: str = None, model_id: str = None):
        """
        Initializes the VLM Client.
        backend_type can be "gemini", "custom_vlm", or "qwen_local".
        """
        self.backend_type = backend_type.lower()
        self.api_key = api_key
        self.api_url = api_url
        self.model_id = model_id

    def generate_caption(self, frame_paths: list[str], context: dict) -> dict:
        """
        Generate a grounded caption for a video segment.
        
        Args:
            frame_paths: List of absolute file paths to the sampled frames of this segment.
            context: {
                "verified_counts": {"car": 3, "person": 1, ...},
                "lighting": "day" | "night",
                "segment_range": "0:00:10 - 0:00:20"
            }
            
        Returns:
            dict containing:
                "caption": str
                "claims": {
                    "objects_mentioned": list of str,
                    "counts_mentioned": dict of class -> count
                }
        """
        # Read frames and convert to base64
        keyframes_b64 = []
        for path in frame_paths:
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                    keyframes_b64.append(b64)
            except Exception as e:
                print(f"[-] Error reading frame {path}: {e}")
                
        if not keyframes_b64:
            return {
                "caption": f"No frames available for segment {context.get('segment_range', '')}.",
                "claims": {"objects_mentioned": [], "counts_mentioned": {}}
            }

        counts_str = ", ".join(f"{count} {cls}(s)" for cls, count in context.get("verified_counts", {}).items())
        lighting = context.get("lighting", "day")
        
        prompt = f"""You are a precise annotation assistant for autonomous driving video datasets.
You are given a chronological sequence of frames from a video segment.
The segment is verified to contain the following object counts: {counts_str or "No objects detected"}.
The scene lighting is: {lighting}.

Your task is to write a brief, 1-2 sentence description (caption) of this segment.
Ensure that:
1. You strictly align with the verified object counts: {counts_str or "No objects"}. Do not mention more or fewer objects than this!
2. You mention the environment and lighting condition (e.g. {lighting}).
3. Avoid generic descriptions. Be specific to what is visible in the frames.

You must respond in JSON format matching the schema:
{{
  "caption": "A brief 1-2 sentence description of the segment.",
  "claims": {{
    "objects_mentioned": ["list", "of", "classes", "mentioned"],
    "counts_mentioned": {{
      "class_name": count_value
    }}
  }}
}}

Ensure class names in "counts_mentioned" are mapped to base classes like "car", "person", "truck", "dog", etc.
Output ONLY the raw JSON object. Do not include markdown code block syntax.
"""
        
        if self.backend_type == "custom_vlm":
            return self._call_custom_vlm(keyframes_b64, prompt)
        elif self.backend_type == "gemini":
            return self._call_gemini(keyframes_b64, prompt)
        elif self.backend_type == "qwen_local":
            return self._call_qwen_local(frame_paths, prompt) # Pass image paths for local processing
        else:
            # Fallback/mock
            return {
                "caption": f"A segment recorded during {lighting} showing: {counts_str}.",
                "claims": {
                    "objects_mentioned": list(context.get("verified_counts", {}).keys()),
                    "counts_mentioned": context.get("verified_counts", {})
                }
            }

    def _call_gemini(self, keyframes_b64: list[str], prompt: str) -> dict:
        if not self.api_key:
            raise ValueError("Gemini API key is required for Gemini VLM backend.")
            
        parts = [{"text": prompt}]
        for b64 in keyframes_b64:
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": b64
                }
            })
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }
        
        try:
            res = requests.post(url, json=payload, timeout=20)
            res.raise_for_status()
            data = res.json()
            text = data["contents"][0]["parts"][0]["text"].strip()
            
            # Clean markdown code block if VLM ignored the generationConfig responseMimeType
            if text.startswith("```json"):
                text = text.split("```json", 1)[1].rsplit("```", 1)[0].strip()
            elif text.startswith("```"):
                text = text.split("```", 1)[1].rsplit("```", 1)[0].strip()
                
            return json.loads(text)
        except Exception as e:
            print(f"[-] Gemini call failed: {e}")
            raise e

    def _call_custom_vlm(self, keyframes_b64: list[str], prompt: str) -> dict:
        if not self.api_url:
            raise ValueError("API URL is required for Custom VLM backend.")
            
        content = [{"type": "text", "text": prompt}]
        for b64 in keyframes_b64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"
                }
            })
            
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        url = self.api_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"
            
        payload = {
            "model": self.model_id or "gpt-4o",
            "messages": [{"role": "user", "content": content}]
        }
        
        try:
            res = requests.post(url, json=payload, headers=headers, timeout=30)
            res.raise_for_status()
            data = res.json()
            text = data["choices"][0]["message"]["content"].strip()
            
            if text.startswith("```json"):
                text = text.split("```json", 1)[1].rsplit("```", 1)[0].strip()
            elif text.startswith("```"):
                text = text.split("```", 1)[1].rsplit("```", 1)[0].strip()
                
            return json.loads(text)
        except Exception as e:
            print(f"[-] Custom VLM call failed: {e}")
            raise e

    def _call_qwen_local(self, frame_paths: list[str], prompt: str) -> dict:
        # Since local Qwen is slow/optional and has import dependencies,
        # we load it conditionally, and fallback to mock if loading fails.
        try:
            import torch
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
            
            # Load model and processor if not already loaded in some global cache
            # For simplicity, we implement it here but note that running local models is optional
            # and Qwen can also be run using custom VLM API URLs.
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_id = self.model_id or "Qwen/Qwen2-VL-2B-Instruct"
            
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map="auto"
            )
            processor = AutoProcessor.from_pretrained(model_id)
            
            content = []
            for path in frame_paths:
                content.append({
                    "type": "image",
                    "image": path,
                })
            content.append({
                "type": "text",
                "text": prompt
            })
            
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                video_kwargs=video_kwargs,
                padding=True,
                return_tensors="pt"
            ).to(device)
            
            generated_ids = model.generate(**inputs, max_new_tokens=150)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            res_text = output_text[0].strip()
            if res_text.startswith("```json"):
                res_text = res_text.split("```json", 1)[1].rsplit("```", 1)[0].strip()
            elif res_text.startswith("```"):
                res_text = res_text.split("```", 1)[1].rsplit("```", 1)[0].strip()
                
            return json.loads(res_text)
        except Exception as e:
            print(f"[-] Local Qwen VLM failed: {e}. Falling back to default mock structure.")
            # Default fallback mock response
            return {
                "caption": "Video segment showing street views.",
                "claims": {"objects_mentioned": [], "counts_mentioned": {}}
            }
