import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import itertools
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Set matplotlib backend for proper display
plt.switch_backend('Agg')  # Use for saving, switch to 'inline' for Jupyter display
class ImageDataset(Dataset):
    """
    Custom Dataset for loading real (A) and anime (B) images.
    Structure expected:
    root/
    trainA/
    trainB/
    testA/
    testB/
    """

    def __init__(self, root, transforms_=None, mode="train", unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(os.listdir(os.path.join(root, f"{mode}A")))
        self.files_B = sorted(os.listdir(os.path.join(root, f"{mode}B")))
        self.root_A = os.path.join(root, f"{mode}A")
        self.root_B = os.path.join(root, f"{mode}B")

    def __getitem__(self, index):
        item_A = self.transform(
            Image.open(os.path.join(self.root_A, self.files_A[index % len(self.files_A)])).convert("RGB")
        )
        if self.unaligned:
            item_B = self.transform(
                Image.open(os.path.join(self.root_B, self.files_B[np.random.randint(0, len(self.files_B))])).convert("RGB")
            )
        else:
            item_B = self.transform(
                Image.open(os.path.join(self.root_B, self.files_B[index % len(self.files_B)])).convert("RGB")
            )
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
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
class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super(Discriminator, self).__init__()
        layers = [
            nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(256, 512, 4, padding=1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(512, 1, 4, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)
class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if np.random.uniform(0, 1) > 0.5:
                    i = np.random.randint(0, self.max_size)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
class CycleGAN:
    def __init__(self, lr=0.0002, b1=0.5, b2=0.999, decay_epoch=100, n_epochs=200):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.netG_A2B = Generator().to(self.device)
        self.netG_B2A = Generator().to(self.device)
        self.netD_A = Discriminator().to(self.device)
        self.netD_B = Discriminator().to(self.device)
        self.netG_A2B.apply(weights_init_normal)
        self.netG_B2A.apply(weights_init_normal)
        self.netD_A.apply(weights_init_normal)
        self.netD_B.apply(weights_init_normal)
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        self.optimizer_G = optim.Adam(
            itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()), lr=lr, betas=(b1, b2)
        )
        self.optimizer_D_A = optim.Adam(self.netD_A.parameters(), lr=lr, betas=(b1, b2))
        self.optimizer_D_B = optim.Adam(self.netD_B.parameters(), lr=lr, betas=(b1, b2))
        self.lr_scheduler_G = optim.lr_scheduler.LambdaLR(
            self.optimizer_G, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step
        )
        self.lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(
            self.optimizer_D_A, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step
        )
        self.lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(
            self.optimizer_D_B, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step
        )
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

    def train_epoch(self, dataloader, epoch, n_epochs):
        self.netG_A2B.train()
        self.netG_B2A.train()
        self.netD_A.train()
        self.netD_B.train()
        for i, batch in enumerate(dataloader):
            real_A = batch["A"].to(self.device)
            real_B = batch["B"].to(self.device)
            valid = torch.ones(real_A.size(0), 1, requires_grad=False, device=self.device)
            fake = torch.zeros(real_A.size(0), 1, requires_grad=False, device=self.device)

            # Train Generators
            self.optimizer_G.zero_grad()
            loss_id_A = self.criterion_identity(self.netG_B2A(real_A), real_A)
            loss_id_B = self.criterion_identity(self.netG_A2B(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2
            fake_B = self.netG_A2B(real_A)
            loss_GAN_A2B = self.criterion_GAN(self.netD_B(fake_B), valid)
            fake_A = self.netG_B2A(real_B)
            loss_GAN_B2A = self.criterion_GAN(self.netD_A(fake_A), valid)
            loss_GAN = (loss_GAN_A2B + loss_GAN_B2A) / 2
            recov_A = self.netG_B2A(fake_B)
            loss_cycle_A = self.criterion_cycle(recov_A, real_A)
            recov_B = self.netG_A2B(fake_A)
            loss_cycle_B = self.criterion_cycle(recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
            loss_G = loss_GAN + 10.0 * loss_cycle + 5.0 * loss_identity
            loss_G.backward()
            self.optimizer_G.step()

            # Train Discriminator A
            self.optimizer_D_A.zero_grad()
            loss_real = self.criterion_GAN(self.netD_A(real_A), valid)
            fake_A_ = self.fake_A_buffer.push_and_pop(fake_A)
            loss_fake = self.criterion_GAN(self.netD_A(fake_A_.detach()), fake)
            loss_D_A = (loss_real + loss_fake) / 2
            loss_D_A.backward()
            self.optimizer_D_A.step()

            # Train Discriminator B
            self.optimizer_D_B.zero_grad()
            loss_real = self.criterion_GAN(self.netD_B(real_B), valid)
            fake_B_ = self.fake_B_buffer.push_and_pop(fake_B)
            loss_fake = self.criterion_GAN(self.netD_B(fake_B_.detach()), fake)
            loss_D_B = (loss_real + loss_fake) / 2
            loss_D_B.backward()
            self.optimizer_D_B.step()

            if i % 100 == 0:
                print(
                    f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {(loss_D_A + loss_D_B).item():.4f}] "
                    f"[G loss: {loss_G.item():.4f}, adv: {loss_GAN.item():.4f}, "
                    f"cycle: {loss_cycle.item():.4f}, identity: {loss_identity.item():.4f}]"
                )

    def train(self, dataloader, n_epochs=200, checkpoint_interval=10):
        os.makedirs("images", exist_ok=True)
        os.makedirs("saved_models", exist_ok=True)
        for epoch in range(n_epochs):
            self.train_epoch(dataloader, epoch, n_epochs)
            self.lr_scheduler_G.step()
            self.lr_scheduler_D_A.step()
            self.lr_scheduler_D_B.step()
            if epoch % 10 == 0:
                print(f"Sampling images at epoch {epoch} (implement saving demo images if needed)")
            if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
                torch.save(self.netG_A2B.state_dict(), f"saved_models/netG_A2B_epoch_{epoch}.pth")
                torch.save(self.netG_B2A.state_dict(), f"saved_models/netG_B2A_epoch_{epoch}.pth")
                print(f"Saved checkpoint at epoch {epoch}")

    def generate_anime(self, real_image_path, output_path):
        self.netG_A2B.eval()
        transform = transforms.Compose([
            transforms.Resize((256, 256), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image = Image.open(real_image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            fake_anime = self.netG_A2B(image)
        fake_anime = 0.5 * (fake_anime + 1.0)
        save_image(fake_anime, output_path)
        print(f"Anime image saved to {output_path}")
def create_dataset(dataset_dir, batch_size=1, img_size=256, mode="train"):
    transforms_ = [
        transforms.Resize((img_size, img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    dataset = ImageDataset(dataset_dir, transforms_=transforms_, unaligned=True, mode=mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader
def visualize_results(cyclegan, real_image_path, save_path=None, device=None):
    """
    FIXED: Display original and anime-style generated images side by side.
    
    Args:
        cyclegan: Trained CycleGAN object
        real_image_path: Path to input image from domain A (real photo)
        save_path: Optional path to save the comparison image
        device: torch.device, defaults to CUDA if available
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set model to evaluation mode
    cyclegan.netG_A2B.eval()

    # Define preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    try:
        # Load and preprocess image
        print(f"Loading image from: {real_image_path}")
        real_image = Image.open(real_image_path).convert('RGB')
        img_input = transform(real_image).unsqueeze(0).to(device)
        print(f"Input image shape: {img_input.shape}")

        # Generate fake image
        with torch.no_grad():
            fake_image = cyclegan.netG_A2B(img_input)
            print(f"Generated image shape: {fake_image.shape}")

        # **CRITICAL FIX**: Proper tensor conversion and denormalization
        if fake_image.requires_grad:
            fake_image = fake_image.detach()
        if fake_image.is_cuda:
            fake_image = fake_image.cpu()
        
        # Denormalize from [-1,1] to [0,1]
        fake_image = (fake_image + 1) * 0.5
        fake_image = torch.clamp(fake_image, 0, 1)
        
        # Convert to numpy array
        fake_image_np = fake_image.squeeze(0).permute(1, 2, 0).numpy()

        # Debug information
        print(f"Fake image range: {fake_image_np.min():.3f} to {fake_image_np.max():.3f}")
        print(f"Fake image shape: {fake_image_np.shape}")
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        axes[0].imshow(real_image)
        axes[0].set_title("Original", fontsize=16)
        axes[0].axis('off')
        
        # Generated anime image
        axes[1].imshow(fake_image_np)
        axes[1].set_title("Anime Style", fontsize=16)
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison saved to: {save_path}")
        
        # Display
        plt.show()
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        import traceback
        traceback.print_exc()

# Alternative function to save individual images
def save_comparison_images(cyclegan, real_image_path, output_dir="output", device=None):
    """Save original and generated images separately"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(output_dir, exist_ok=True)
    
    cyclegan.netG_A2B.eval()
    transform = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Load image
    real_image = Image.open(real_image_path).convert('RGB')
    img_input = transform(real_image).unsqueeze(0).to(device)
    
    # Generate
    with torch.no_grad():
        fake_image = cyclegan.netG_A2B(img_input)
    
    # Save using torchvision
    fake_image_norm = (fake_image + 1) * 0.5  # Denormalize to [0,1]
    
    # Save images
    original_path = os.path.join(output_dir, "original.jpg")
    generated_path = os.path.join(output_dir, "anime_generated.jpg")
    
    real_image.save(original_path)
    save_image(fake_image_norm, generated_path)
    
    print(f"Original saved to: {original_path}")
    print(f"Generated saved to: {generated_path}")
    
    return original_path, generated_path
# if __name__ == "__main__":
#     # Configuration
#     DATASET_DIR = "/kaggle/input/selfie2anime"
#     BATCH_SIZE = 5 
#     N_EPOCHS = 40
    
#     print("PyTorch CycleGAN for Real↔Anime Face Conversion")
#     print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
#     # Create dataset
#     dataloader = create_dataset(DATASET_DIR, batch_size=BATCH_SIZE, mode="train")
    
#     # Initialize CycleGAN
#     cyclegan = CycleGAN()
#     total_params_G = sum(p.numel() for p in cyclegan.netG_A2B.parameters())
#     total_params_D = sum(p.numel() for p in cyclegan.netD_A.parameters())
#     print(f"Generator parameters: {total_params_G:,}")
#     print(f"Discriminator parameters: {total_params_D:,}")
    
#     # Train the model
#     cyclegan.train(dataloader, n_epochs=N_EPOCHS)
    
#     print("Training completed!")
# Test the visualization with a sample image
def test_visualization():
    # You need to provide a path to a test image
    test_image_path = "/kaggle/input/selfie2anime/testA/1.jpg"  # Update this path
    
    if os.path.exists(test_image_path):
        print("Testing visualization...")
        try:
            # Test the fixed visualization function
            visualize_results(cyclegan, test_image_path, save_path="comparison.png")
            
            # Alternative: Save individual images
            save_comparison_images(cyclegan, test_image_path, output_dir="test_output")
            
        except Exception as e:
            print(f"Visualization test failed: {e}")
    else:
        print(f"Test image not found at: {test_image_path}")
        print("Please update the test_image_path variable with a valid image path")

# Uncomment to test visualization
# test_visualization()
def load_cyclegan_generator(model_path, device=None):
    """
    Load a pre-trained CycleGAN generator from saved weights
    
    Args:
        model_path: Path to the saved model weights (.pth file)
        device: torch.device, if None will auto-detect CUDA/CPU
    
    Returns:
        generator: Loaded Generator model ready for inference
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from: {model_path}")
    print(f"Using device: {device}")
    
    # Initialize generator
    generator = Generator()
    
    # Load the saved weights
    try:
        if device.type == 'cuda':
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location='cpu')
        
        generator.load_state_dict(state_dict)
        generator.to(device)
        generator.eval()  # Set to evaluation mode
        
        print("✓ Model loaded successfully!")
        return generator
        
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return None
def visualize_with_loaded_model(model_path, input_image_path, output_dir="visualization_output", save_comparison=True, show_plot=True):
    """
    Complete visualization function that loads model and generates comparisons
    
    Args:
        model_path: Path to saved generator model (.pth file)
        input_image_path: Path to input image
        output_dir: Directory to save outputs
        save_comparison: Whether to save comparison image
        show_plot: Whether to display the plot (set False for headless environments)
    """
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    generator = load_cyclegan_generator(model_path, device)
    if generator is None:
        print("Failed to load model. Exiting.")
        return None
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        # Load and preprocess input image
        print(f"Loading input image: {input_image_path}")
        original_image = Image.open(input_image_path).convert('RGB')
        input_tensor = transform(original_image).unsqueeze(0).to(device)
        
        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Input tensor range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
        
        # Generate anime-style image
        with torch.no_grad():
            fake_tensor = generator(input_tensor)
        
        print(f"Generated tensor shape: {fake_tensor.shape}")
        print(f"Generated tensor range: [{fake_tensor.min():.3f}, {fake_tensor.max():.3f}]")
        
        # Process generated image for visualization
        # Move to CPU and detach from computation graph
        fake_image = fake_tensor.detach().cpu()
        
        # Denormalize from [-1, 1] to [0, 1]
        fake_image = (fake_image + 1.0) / 2.0
        fake_image = torch.clamp(fake_image, 0, 1)
        
        # Convert to numpy array for matplotlib
        fake_image_np = fake_image.squeeze(0).permute(1, 2, 0).numpy()
        
        print(f"Final image range: [{fake_image_np.min():.3f}, {fake_image_np.max():.3f}]")
        
        # Create and save comparison
        if save_comparison or show_plot:
            # Set up matplotlib for different environments
            if not show_plot:
                plt.switch_backend('Agg')  # Non-interactive backend
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 7))
            
            # Original image
            axes[0].imshow(original_image)
            axes[0].set_title("Original Image", fontsize=16, fontweight='bold')
            axes[0].axis('off')
            
            # Generated anime image
            axes[1].imshow(fake_image_np)
            axes[1].set_title("Generated Anime Style", fontsize=16, fontweight='bold')
            axes[1].axis('off')
            
            plt.tight_layout()
            
            # Save comparison
            if save_comparison:
                comparison_path = os.path.join(output_dir, "comparison.png")
                plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
                print(f"✓ Comparison saved: {comparison_path}")
            
            # Show plot if requested
            if show_plot:
                plt.show()
            else:
                plt.close()
        
        # Save individual images
        original_path = os.path.join(output_dir, "original.jpg")
        generated_path = os.path.join(output_dir, "generated_anime.jpg")
        
        # Save original
        original_image.save(original_path, quality=95)
        
        # Save generated using torchvision (maintains quality)
        save_image(fake_image, generated_path)
        
        print(f"✓ Original saved: {original_path}")
        print(f"✓ Generated saved: {generated_path}")
        
        return {
            'original_path': original_path,
            'generated_path': generated_path,
            'comparison_path': comparison_path if save_comparison else None,
            'generator': generator  # Return loaded model for further use
        }
        
    except Exception as e:
        print(f"✗ Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
def batch_visualize_with_loaded_model(model_path, input_folder, output_folder="batch_output"):
    """
    Process multiple images with loaded model
    
    Args:
        model_path: Path to saved generator model
        input_folder: Folder containing input images
        output_folder: Folder to save all outputs
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_folder, exist_ok=True)
    
    # Load model once
    generator = load_cyclegan_generator(model_path, device)
    if generator is None:
        return
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Find all images in input folder
    image_files = [f for f in os.listdir(input_folder) 
                   if os.path.splitext(f.lower())[1] in image_extensions]
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    results = []
    for i, image_file in enumerate(image_files, 1):
        print(f"\nProcessing {i}/{len(image_files)}: {image_file}")
        
        input_path = os.path.join(input_folder, image_file)
        output_dir = os.path.join(output_folder, f"result_{os.path.splitext(image_file)[0]}")
        
        result = visualize_with_loaded_model(
            model_path=model_path,
            input_image_path=input_path,
            output_dir=output_dir,
            save_comparison=True,
            show_plot=False  # Don't show plots in batch mode
        )
        
        if result:
            results.append(result)
            print(f"✓ Completed: {image_file}")
        else:
            print(f"✗ Failed: {image_file}")
    
    print(f"\nBatch processing completed! {len(results)}/{len(image_files)} successful")
    return results
if __name__ == "__main__":
    # Update these paths according to your setup
    MODEL_PATH = "netG_A2B_epoch_20 (1).pth"  # Path to your saved model
    INPUT_IMAGE = "360_F_611543432_8GkBOGNH9QeRsIe4ZhQMraDEzz4L5M13.jpg"  # Path to test image
    
    # Test if files exist
    if os.path.exists(MODEL_PATH) and os.path.exists(INPUT_IMAGE):
        print("Starting visualization with loaded model...")
        
        result = visualize_with_loaded_model(
            model_path=MODEL_PATH,
            input_image_path=INPUT_IMAGE,
            output_dir="final_results",
            save_comparison=True,
            show_plot=True  # Set to False if running in headless environment
        )
        
        if result:
            print("✓ Visualization completed successfully!")
            print(f"Check the 'final_results' folder for outputs")
        else:
            print("✗ Visualization failed")
    else:
        print("Model or input image not found!")
        print(f"Model path: {MODEL_PATH} (exists: {os.path.exists(MODEL_PATH)})")
        print(f"Input image: {INPUT_IMAGE} (exists: {os.path.exists(INPUT_IMAGE)})")

# Example 2: Batch processing
# batch_results = batch_visualize_with_loaded_model(
#     model_path="saved_models/netG_A2B_epoch_10.pth",
#     input_folder="/kaggle/input/selfie2anime/testA",
#     output_folder="batch_anime_results"
# )
