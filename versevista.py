import base64
import requests
from PIL import Image
import io
import os
from typing import Optional, Dict, Any

class VerseVista:
    def __init__(self, api_provider: str = "openai", api_key: str = None):
        """
        Initialize VerseVista with your preferred API provider
        
        Args:
            api_provider: "openai", "anthropic", or "google"
            api_key: Your API key
        """
        self.api_provider = api_provider.lower()
        self.api_key = api_key or os.getenv(f"{api_provider.upper()}_API_KEY")
        
        if not self.api_key:
            raise ValueError(f"API key required for {api_provider}")
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def resize_image_if_needed(self, image_path: str, max_size: tuple = (1024, 1024)) -> str:
        """Resize image if it's too large (for API limits)"""
        with Image.open(image_path) as img:
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Save resized image temporarily
                temp_path = f"temp_resized_{os.path.basename(image_path)}"
                img.save(temp_path, format='JPEG', quality=85)
                return temp_path
        return image_path
    
    def generate_poem_openai(self, image_path: str, style: str = "free verse") -> str:
        """Generate poem using OpenAI GPT-4V"""
        # Resize if needed
        processed_image = self.resize_image_if_needed(image_path)
        base64_image = self.encode_image_to_base64(processed_image)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": "gpt-4o",  # or "gpt-4-vision-preview"
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Analyze this image and write a beautiful {style} poem inspired by what you see. 
                            Consider the mood, colors, subjects, emotions, and atmosphere in the image. 
                            Make the poem creative, evocative, and emotionally resonant. 
                            Aim for 8-16 lines."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300,
            "temperature": 0.8
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", 
            headers=headers, 
            json=payload
        )
        
        # Clean up temp file if created
        if processed_image != image_path:
            os.remove(processed_image)
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    def generate_poem_anthropic(self, image_path: str, style: str = "free verse") -> str:
        """Generate poem using Anthropic Claude"""
        processed_image = self.resize_image_if_needed(image_path)
        base64_image = self.encode_image_to_base64(processed_image)
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 300,
            "temperature": 0.8,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": f"""Look at this image and write a beautiful {style} poem inspired by what you see. 
                            Capture the essence, mood, and emotions evoked by the visual elements. 
                            Be creative and evocative. Write 8-16 lines."""
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload
        )
        
        # Clean up temp file if created
        if processed_image != image_path:
            os.remove(processed_image)
        
        if response.status_code == 200:
            return response.json()['content'][0]['text']
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    def generate_poem(self, image_path: str, style: str = "free verse", 
                     theme: Optional[str] = None) -> Dict[str, Any]:
        """
        Main method to generate poem from image
        
        Args:
            image_path: Path to the input image
            style: Poetry style ("free verse", "sonnet", "haiku", "rhyming", etc.)
            theme: Optional theme to guide the poem
        
        Returns:
            Dictionary with poem text and metadata
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Add theme to style if provided
        if theme:
            style_prompt = f"{style} poem with a {theme} theme"
        else:
            style_prompt = style
        
        try:
            if self.api_provider == "openai":
                poem_text = self.generate_poem_openai(image_path, style_prompt)
            elif self.api_provider == "anthropic":
                poem_text = self.generate_poem_anthropic(image_path, style_prompt)
            else:
                raise ValueError(f"Unsupported API provider: {self.api_provider}")
            
            return {
                "poem": poem_text.strip(),
                "image_path": image_path,
                "style": style,
                "theme": theme,
                "api_provider": self.api_provider,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "poem": None,
                "error": str(e),
                "status": "failed"
            }


# Usage example
def main():
    api_key=os.getenv('OPENAI_API_KEY')
    # Initialize with your API key
    verse_vista = VerseVista(
        api_provider="openai",  # or "anthropic"
        api_key=str(api_key)
    )
    
    # Generate poem from image
    result = verse_vista.generate_poem(
        image_path="./image/wally_bg_77.jpg",
        style="free verse",
        theme="nature"  # optional
    )
    
    if result["status"] == "success":
        print("Generated Poem:")
        print("-" * 40)
        print(result["poem"])
    else:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()

