import os
import argparse
from data_loading import PoemDataLoader, get_coco_datasets, create_data_loaders
from versevista import VerseVista  # Our API-based solution


def demo_api_approach(image_path: str, api_provider: str = "openai"):
    """Demonstrate the API-based approach"""
    print("🎨 VerseVista - API Approach Demo")
    print("=" * 50)
    
    # Get API key from environment
    if api_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    elif api_provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    else:
        raise ValueError("Unsupported API provider")
    
    if not api_key:
        print(f"❌ {api_provider.upper()}_API_KEY not found in environment variables")
        print(f"Please set your API key: export {api_provider.upper()}_API_KEY='your-key-here'")
        return
    
    # Initialize VerseVista
    verse_vista = VerseVista(api_provider=api_provider, api_key=api_key)
    
    # Test different styles
    styles = ["free verse", "haiku", "rhyming"]
    
    for style in styles:
        print(f"\n📝 Generating {style} poem...")
        result = verse_vista.generate_poem(
            image_path=image_path,
            style=style,
            theme="nature"
        )
        
        if result["status"] == "success":
            print(f"\n✨ {style.title()} Poem:")
            print("-" * 30)
            print(result["poem"])
            print("-" * 30)
        else:
            print(f"❌ Error: {result['error']}")


def demo_custom_models():
    """Demonstrate the custom model approach (if you want to continue with it)"""
    print("\n🔧 Custom Models Demo")
    print("=" * 50)
    
    # Check if poem dataset exists
    poem_dir = "poem_dataset"
    if not os.path.exists(poem_dir):
        print(f"❌ Poem directory '{poem_dir}' not found")
        print("Create this directory and add some .txt files with poems to test")
        return
    
    # Load poem dataset
    poem_dataset = PoemDataLoader(poem_dir)
    print(f"📚 Loaded {len(poem_dataset)} poems")
    
    if len(poem_dataset) > 0:
        print(f"📖 Sample poem preview:")
        sample_poem = poem_dataset[0]
        print(f"'{sample_poem[:100]}...'")
    
    # Check COCO dataset
    print("\n🖼️ Checking COCO dataset...")
    coco_train, coco_val = get_coco_datasets()
    
    if coco_train:
        print(f"✅ COCO training set: {len(coco_train)} images")
        print(f"✅ COCO validation set: {len(coco_val)} images")
    else:
        print("❌ COCO dataset not found")
        print("Download from: https://cocodataset.org/#download")


def create_sample_datasets():
    """Create sample datasets for testing"""
    print("📁 Creating sample datasets...")
    
    # Create poem dataset directory
    poem_dir = "poem_dataset"
    os.makedirs(poem_dir, exist_ok=True)
    
    # Sample poems
    sample_poems = [
        """The golden sun sets low behind the hills,
Painting the sky in orange and pink hues.
A gentle breeze through autumn leaves spills,
As twilight brings the day's end news.""",
        
        """Moonlight dances on the water's face,
Silver ripples catch the night's embrace.
Stars above in silent conversation,
Nature's own divine creation.""",
        
        """In the garden where roses bloom,
Petals soft dispel all gloom.
Fragrance sweet fills morning air,
Beauty found beyond compare."""
    ]
    
    for i, poem in enumerate(sample_poems, 1):
        with open(f"{poem_dir}/sample_poem_{i}.txt", "w") as f:
            f.write(poem)
    
    print(f"✅ Created {len(sample_poems)} sample poems in '{poem_dir}'")


def main():
    parser = argparse.ArgumentParser(description="VerseVista - Generate poetry from images")
    parser.add_argument("--mode", choices=["api", "custom", "setup"], default="api",
                       help="Choose demo mode")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai",
                       help="API provider for poem generation")
    
    args = parser.parse_args()
    
    if args.mode == "setup":
        create_sample_datasets()
        return
    
    if args.mode == "api":
        if not args.image:
            print("❌ Please provide an image path with --image")
            print("Example: python main.py --mode api --image path/to/your/image.jpg")
            return
        
        if not os.path.exists(args.image):
            print(f"❌ Image file not found: {args.image}")
            return
        
        demo_api_approach(args.image, args.provider)
    
    elif args.mode == "custom":
        demo_custom_models()


if __name__ == "__main__":
    # If no arguments provided, show help
    import sys
    if len(sys.argv) == 1:
        print("🎨 VerseVista - AI Poetry Generator")
        print("=" * 40)
        print("\nUsage examples:")
        print("1. Set up sample data:")
        print("   python main.py --mode setup")
        print("\n2. Generate poem from image (API):")
        print("   python main.py --mode api --image your_image.jpg")
        print("\n3. Test custom models:")
        print("   python main.py --mode custom")
        print("\n4. Run web interface:")
        print("   streamlit run streamlit_app.py")
        print("\nMake sure to set your API keys as environment variables!")
    else:
        main()

