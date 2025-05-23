import streamlit as st
import tempfile
import os
from PIL import Image
from versevista import VerseVista

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(
    page_title="VerseVista - AI Poetry Generator",
    page_icon="🎨",
    layout="wide"
)

def get_api_key(provider):
    """Get API key from Streamlit secrets or environment variables"""
    key_name = f"{provider.upper()}_API_KEY"
    
    try:
        return st.secrets[key_name]
    except (KeyError, FileNotFoundError):
        return os.getenv(key_name)

def main():
    st.title("🎨 VerseVista")
    st.subheader("Transform Your Images into Beautiful Poetry")
    
    # Create two columns
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("### 📸 Upload Your Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg', 'webp', 'bmp', 'gif'],
            help="Upload an image that inspires you"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Your uploaded image", use_column_width=True)
            
            # Image info
            st.info(f"📊 Image size: {image.size[0]} × {image.size[1]} pixels")
    
    with col2:
        st.markdown("### ⚙️ Settings")
        
        # API Provider selection
        api_provider = st.selectbox(
            "Choose AI Provider",
            ["openai", "anthropic"],
            help="Select which AI service to use for poem generation"
        )
        
        # Poetry style selection
        poetry_style = st.selectbox(
            "Poetry Style",
            [
                "free verse",
                "rhyming", 
                "haiku",
                "sonnet",
                "limerick",
                "blank verse",
                "prose poem"
            ]
        )
        
        # Optional theme
        theme = st.text_input(
            "Theme (optional)",
            placeholder="e.g., nature, love, melancholy",
            help="Optional theme to guide the poem's direction"
        )
        
        # Generate button
        if uploaded_file is not None:
            if st.button("✨ Generate Poem", type="primary", use_container_width=True):
                # Get API key using the new function
                api_key = get_api_key(api_provider)
                
                if not api_key:
                    st.error(f"❌ {api_provider.upper()}_API_KEY not found. Please check your secrets configuration.")
                    st.info("💡 **For Streamlit Cloud**: Add your API key in the app settings under 'Secrets management'")
                    st.info("💡 **For local development**: Add your API key to a .env file")
                    st.stop()
                
                with st.spinner("🎭 Crafting your poem..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Initialize VerseVista
                        verse_vista = VerseVista(
                            api_provider=api_provider,
                            api_key=api_key
                        )
                        
                        # Generate poem
                        result = verse_vista.generate_poem(
                            image_path=tmp_path,
                            style=poetry_style,
                            theme=theme if theme else None
                        )
                        
                        # Clean up temp file
                        os.unlink(tmp_path)
                        
                        if result["status"] == "success":
                            st.success("🎉 Poem generated successfully!")
                            
                            # Display the poem
                            st.markdown("### 📝 Your Generated Poem:")
                            
                            # Create a nice poem display
                            poem_lines = result["poem"].split('\n')
                            poem_html = "<div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #007bff;'>"
                            
                            for line in poem_lines:
                                if line.strip():
                                    poem_html += f"<p style='margin: 5px 0; font-style: italic; color: #2c3e50;'>{line}</p>"
                                else:
                                    poem_html += "<br>"
                            
                            poem_html += "</div>"
                            st.markdown(poem_html, unsafe_allow_html=True)
                            
                            # Poem metadata
                            with st.expander("📋 Poem Details"):
                                st.write(f"**Style:** {result['style']}")
                                if result['theme']:
                                    st.write(f"**Theme:** {result['theme']}")
                                st.write(f"**Generated by:** {result['api_provider'].title()}")
                            
                            # Download option
                            poem_text = result["poem"]
                            st.download_button(
                                label="📥 Download Poem",
                                data=poem_text,
                                file_name=f"versevista_poem_{poetry_style.replace(' ', '_')}.txt",
                                mime="text/plain"
                            )
                            
                        else:
                            st.error(f"❌ Error generating poem: {result['error']}")
                            
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {str(e)}")
                        if 'tmp_path' in locals():
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
        else:
            st.info("👆 Please upload an image to generate a poem")
    
    # Footer with tips
    st.markdown("---")
    with st.expander("💡 Tips for Better Poems"):
        st.markdown("""
        - **Clear images work best**: Photos with clear subjects and good lighting tend to generate more vivid poetry
        - **Experiment with styles**: Try different poetry styles to see what works best with your image
        - **Use themes**: Adding a theme can help guide the AI toward the mood you want
        - **Try different providers**: OpenAI and Anthropic may have different creative approaches
        """)

if __name__ == "__main__":
    main()
