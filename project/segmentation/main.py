# main.py

from data_loader import get_dataloader
from visualize import visualize_index

def main():
    image_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\semantic_segmentation\images'
    mask_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\semantic_segmentation\masks'
    json_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\semantic_segmentation\json'

    # Initialize dataloader
    dataloader = get_dataloader(image_dir, mask_dir, json_dir, batch_size=4, num_workers=2)

    # Visualize the first index
    visualize_index(dataloader, 0)  # Change index as needed

if __name__ == "__main__":
    main()
