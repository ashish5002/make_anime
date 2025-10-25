import streamlit as st
import torch
import torch.nn as nn
from torchvision.utils import save_image
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import gdown

# -------- Custom CSS for Beautiful UI -------- #
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');
    
    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2.5rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.95);
        margin-top: 0.5rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #333;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        text-align: center;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        width: 100%;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.4) !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(17, 153, 142, 0.6) !important;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 2rem;
        border: 2px dashed #667eea;
    }
    
    [data-testid="stFileUploader"] section {
        border: none !important;
        background-color: transparent !important;
    }
    
    /* Info/Success/Warning boxes */
    .stAlert {
        border-radius: 10px !important;
        border-left: 4px solid #667eea !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: white;
    }
    
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    [data-testid="stSidebar"] h4 {
        color: white !important;
    }
    
    [data-testid="stSidebar"] p {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.2) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 255, 255, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
    }
    
    /* Image styling */
    [data-testid="stImage"] {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Success messages */
    .stSuccess {
        background-color: #d4edda !important;
        border-color: #c3e6cb !important;
        color: #155724 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
    
    /* Upload area text */
    .uploadedFileName {
        color: #667eea !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# -------- CycleGAN Generator Definition -------- #

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_blocks=9):
        super(Generator, self).__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        in_feats, out_feats = 64, 128
        for _ in range(2):
            layers += [
                nn.Conv2d(in_feats, out_feats, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_feats),
                nn.ReLU(inplace=True)]
            in_feats, out_feats = out_feats, out_feats * 2
        for _ in range(n_blocks):
            layers += [ResidualBlock(in_feats)]
        out_feats = in_feats // 2
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(in_feats, out_feats, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_feats),
                nn.ReLU(inplace=True)]
            in_feats, out_feats = out_feats, out_feats // 2
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# -------- Google Drive Download -------- #

def download_from_gdrive(gdrive_url, output_path):
    """Download file from Google Drive"""
    try:
        if '/file/d/' in gdrive_url:
            file_id = gdrive_url.split('/file/d/')[1].split('/')[0]
        elif 'id=' in gdrive_url:
            file_id = gdrive_url.split('id=')[1].split('&')[0]
        else:
            file_id = gdrive_url
        
        download_url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(download_url, output_path, quiet=False)
        return True
    except Exception as e:
        st.error(f"Error downloading from Google Drive: {str(e)}")
        return False

# -------- Model Loading (Backend) -------- #

@st.cache_resource()
def load_generator_from_gdrive(gdrive_url, model_name="model.pth", device='cpu'):
    """Download and load generator from Google Drive"""
    model_path = model_name
    
    if not os.path.exists(model_path):
        st.info("🔄 Downloading model from Google Drive...")
        with st.spinner("Downloading... This may take a few minutes."):
            if not download_from_gdrive(gdrive_url, model_path):
                st.error("❌ Failed to download model!")
                st.stop()
        st.success("✅ Model downloaded successfully!")
    
    model = Generator()
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, model_path
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# -------- Inference -------- #

def generate_anime_style(generator, input_image, device='cpu'):
    """Convert image to anime style"""
    transform = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    img_tensor = transform(input_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_tensor = generator(img_tensor)
    
    output_tensor = (output_tensor + 1) / 2
    output_tensor = torch.clamp(output_tensor, 0, 1)
    output_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    return output_np, output_tensor

def create_comparison(original_img, anime_np):
    """Create side-by-side comparison with beautiful styling"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('#ffffff')
    
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=18, fontweight='bold', 
                      color='#333', pad=20, fontfamily='sans-serif')
    axes[0].axis('off')
    
    axes[1].imshow(anime_np)
    axes[1].set_title("Anime Style ✨", fontsize=18, fontweight='bold', 
                      color='#667eea', pad=20, fontfamily='sans-serif')
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig

def tensor_to_pil(tensor):
    """Convert tensor to PIL Image"""
    img_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    return Image.fromarray(img_np)

# -------- Streamlit UI -------- #

st.set_page_config(
    page_title="✨ Anime Style Converter",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_custom_css()

# -------- CONFIGURATION -------- #
GDRIVE_URL = "https://drive.google.com/file/d/1542Jk8Yra9ZmfZSIocEiJtn0s-fnsykt/view?usp=drive_link"
MODEL_NAME = "netG.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------- Sidebar -------- #
with st.sidebar:
    st.markdown("### 🎨 Anime Converter")
    st.markdown("---")
    
    st.markdown("#### ℹ️ About")
    st.markdown("""
    Transform your photos into beautiful anime-style artwork using state-of-the-art CycleGAN technology.
    """)
    
    st.markdown("---")
    st.markdown("#### ⚙️ System Info")
    st.markdown(f"**Device:** `{device.upper()}`")
    st.markdown(f"**Model:** `{MODEL_NAME}`")
    
    if os.path.exists(MODEL_NAME):
        file_size = os.path.getsize(MODEL_NAME) / (1024 * 1024)
        st.success(f"✅ Model Ready ({file_size:.1f} MB)")
    else:
        st.warning("⏳ Model will download on first use")
    
    st.markdown("---")
    
    if st.button("🔄 Refresh Model"):
        if os.path.exists(MODEL_NAME):
            os.remove(MODEL_NAME)
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("#### 💡 Pro Tips")
    st.markdown("""
    - Use clear, front-facing photos
    - Good lighting = better results
    - Female portraits work best
    - Neutral expressions preferred
    - Avoid extreme angles
    """)

# -------- Main Content -------- #

# Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">✨ Anime Style Converter</h1>
    <p class="subtitle">Transform your photos into stunning anime artwork with AI</p>
</div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("🎨 Loading AI model..."):
    generator, model_path = load_generator_from_gdrive(GDRIVE_URL, MODEL_NAME, device=device)

# File uploader
st.markdown("### 📤 Upload Your Image")
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=['png', 'jpg', 'jpeg', 'bmp'],
    help="Supported formats: PNG, JPG, JPEG, BMP"
)

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    
    # Display original
    st.markdown("---")
    st.markdown('<div class="section-header">📸 Your Original Image</div>', unsafe_allow_html=True)
    st.image(original_image, use_column_width=True, caption="Original Image")
    
    # Generate button
    st.markdown("")
    if st.button("✨ Transform to Anime Style", type="primary"):
        with st.spinner("🎨 Creating your anime masterpiece..."):
            anime_np, anime_tensor = generate_anime_style(generator, original_image, device=device)
            
            st.balloons()
            st.success("🎉 Transformation Complete!")
            
            # Comparison
            st.markdown("---")
            st.markdown('<div class="section-header">🔄 Before & After Comparison</div>', unsafe_allow_html=True)
            fig = create_comparison(original_image, anime_np)
            st.pyplot(fig)
            
            # Result
            st.markdown("---")
            st.markdown('<div class="section-header">✨ Your Anime Masterpiece</div>', unsafe_allow_html=True)
            st.image(anime_np, use_column_width=True, caption="Anime Style Result")
            
            # Download
            st.markdown("---")
            st.markdown('<div class="section-header">💾 Save Your Creation</div>', unsafe_allow_html=True)
            
            result_pil = tensor_to_pil(anime_tensor)
            buf = io.BytesIO()
            result_pil.save(buf, format='PNG')
            byte_im = buf.getvalue()
            
            st.download_button(
                label="⬇️ Download Anime Image (PNG)",
                data=byte_im,
                file_name="anime_style_result.png",
                mime="image/png"
            )

else:
    # Welcome message
    st.markdown("---")
    st.info("👆 Upload an image above to begin your transformation!")
    
    with st.expander("📖 How to get the best results"):
        st.markdown("""
        **Best Practices:**
        - ✅ Use clear, high-quality photos
        - ✅ Front-facing portraits work best
        - ✅ Ensure good lighting conditions
        - ✅ Neutral or gentle expressions
        - ✅ Female portraits with open hair
        - ❌ Avoid extreme angles or occlusions
        - ❌ Avoid images with visible teeth/wide smiles
        
        **Image Requirements:**
        - Resized to 256x256 for processing
        - Accepts PNG, JPG, JPEG, BMP formats
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #666;'>
    <p><strong>Built with ❤️ using PyTorch & Streamlit | Powered by CycleGAN</strong></p>
    <p style='font-size: 0.9rem;'>Transform Reality into Art</p>
</div>
""", unsafe_allow_html=True)
