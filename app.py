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

# -------- Model Loading (Backend) -------- #

@st.cache_resource()
def load_generator_from_path(model_path, device='cpu'):
    """Load generator from backend file path"""
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    
    model = Generator()
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
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
    
    # Denormalize
    output_tensor = (output_tensor + 1) / 2
    output_tensor = torch.clamp(output_tensor, 0, 1)
    
    # Convert to numpy for display


    output_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    return output_np, output_tensor

#visualization 

def create_comparison(original_img, anime_np):
    """Create side-by-side comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(anime_np)
    axes[1].set_title("Anime Style", fontsize=14, fontweight='bold')
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
    page_title="CycleGAN Anime Converter",
    page_icon="🎨",
    layout="centered"
)

# Header
st.title("🎨 CycleGAN Anime Face Converter")
st.markdown("""
Convert your photos to anime style using CycleGAN!  
Upload an image and get an anime-styled version instantly.
""")

# -------- CONFIGURATION: Set your model path here -------- #
MODEL_PATH = "netG_A2B_epoch_20 (1).pth"  # Change this to your model path
# --------------------------------------------------------- #

# Device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    This app uses a pre-trained CycleGAN model to convert 
    real photos into anime-style images.
    """)
    st.markdown(f"**Device:** `{device.upper()}`")
    st.markdown(f"**Model:** `{os.path.basename(MODEL_PATH)}`")
    
    if st.checkbox("Show Model Info"):
        if os.path.exists(MODEL_PATH):
            file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            st.success(f"✓ Model loaded ({file_size:.2f} MB)")
        else:
            st.error("✗ Model file not found!")

# Load model (cached)
with st.spinner("Loading model..."):
    generator = load_generator_from_path(MODEL_PATH, device=device)
    st.success("✓ Model loaded successfully!")

# File uploader
uploaded_file = st.file_uploader(
    "Upload your image",
    type=['png', 'jpg', 'jpeg', 'bmp'],
    help="Supported formats: PNG, JPG, JPEG, BMP"
)

if uploaded_file is not None:
    # Load image
    original_image = Image.open(uploaded_file).convert("RGB")
    
    # Display original
    st.subheader("Original Image")
    st.image(original_image, use_container_width=True)
    
    # Generate button
    if st.button("🎨 Generate Anime Style", type="primary"):
        with st.spinner("Generating anime style..."):
            # Generate
            anime_np, anime_tensor = generate_anime_style(
                generator, 
                original_image, 
                device=device
            )
            
            st.success("✓ Conversion complete!")
            
            # Show comparison
            st.subheader("Comparison")
            fig = create_comparison(original_image, anime_np)
            st.pyplot(fig)
            
            # Show anime result
            st.subheader("Anime Style Result")
            st.image(anime_np, use_container_width=True)
            
            # Download button
            st.subheader("Download Result")
            
            # Convert tensor to PIL for download
            result_pil = tensor_to_pil(anime_tensor)
            
            # Save to bytes
            buf = io.BytesIO()
            result_pil.save(buf, format='PNG')
            byte_im = buf.getvalue()
            
            st.download_button(
                label="⬇️ Download Anime Image",
                data=byte_im,
                file_name="anime_result.png",
                mime="image/png"
            )

else:
    st.info("👆 Upload an image to get started!")
    
    # Optional: Show example
    with st.expander("📝 Tips for best results"):
        st.markdown("""
        - Use clear, front-facing photos
        - Images with faces work best
        - Good lighting improves results
        - The model works on 256x256 images
        - Make sure u are a girl 
        - Don't show your teeth
        """)

# Footer
st.markdown("---")
st.caption("Built with PyTorch and Streamlit | CycleGAN for Real↔Anime Conversion")
