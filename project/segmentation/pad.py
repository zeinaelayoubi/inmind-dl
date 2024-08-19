from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def pad_image(image, target_size):
    # Original image size
    orig_width, orig_height = image.size
    
    # Calculate the new size while maintaining the aspect ratio
    aspect_ratio = orig_width / orig_height
    
    if orig_width > orig_height:
        new_width = target_size[0]
        new_height = int(target_size[0] / aspect_ratio)
    else:
        new_height = target_size[1]
        new_width = int(target_size[1] * aspect_ratio)
    
    # Resize the image while maintaining the aspect ratio
    image_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create a new image with the target size and padding
    new_image = Image.new('RGB', target_size, (0, 0, 0))  # Black padding
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    new_image.paste(image_resized, (paste_x, paste_y))
    
    return new_image

# Define the transform pipeline
transform = transforms.Compose([
    transforms.Lambda(lambda img: pad_image(img, (256, 256))),  # Apply padding
    transforms.ToTensor()
])

# Example usage
image_path = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\semantic_segmentation\train\images\rgb_0946.png'
image = Image.open(image_path)
transformed_image = transform(image)

# Convert tensor back to PIL image
def tensor_to_pil(tensor):
    tensor = tensor.mul(255).byte()
    tensor = tensor.permute(1, 2, 0)  # Convert from CxHxW to HxWxC
    return Image.fromarray(tensor.numpy())

# Convert and display
padded_image = tensor_to_pil(transformed_image)
plt.imshow(padded_image)
plt.axis('off')  # Hide the axes
plt.show()
