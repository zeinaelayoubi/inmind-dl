import requests

# URLs for your FastAPI endpoints
BASE_URL = "http://127.0.0.1:8000"

# File path
IMAGE_PATH = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\object_detection\images\rgb_0001.png'

def test_segment():
    url = f"{BASE_URL}/segment"
    with open(IMAGE_PATH, "rb") as image_file:
        response = requests.post(url, files={"file": image_file})
        if response.status_code == 200:
            with open("segmentation_result.png", "wb") as result_file:
                result_file.write(response.content)
            print("Segmentation image saved successfully.")
        else:
            print(f"Error {response.status_code}: {response.text}")

def test_detect():
    url = f"{BASE_URL}/detect"
    with open(IMAGE_PATH, "rb") as image_file:
        response = requests.post(url, files={"file": image_file})
        if response.status_code == 200:
            with open("detection_result.png", "wb") as result_file:
                result_file.write(response.content)
            print("Detection image saved successfully.")
        else:
            print(f"Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    test_segment()
    test_detect()
