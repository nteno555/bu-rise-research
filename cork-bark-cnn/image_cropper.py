import os
from PIL import Image


def process_images():
    input_folder = "/Users/jennifer/Documents/BU RISE/bu-rise-research/cork-bark-cnn/cork_oak_dataset/Training/0_Healthy"
    output_folder = "/Users/jennifer/Documents/BU RISE/bu-rise-research/cork-bark-cnn/cork_oak_dataset/Training/0_Healthy_Processed"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    valid_extensions = ('.jpg', '.jpeg', '.JPG', '.JPEG')

    files = [f for f in os.listdir(input_folder) if f.endswith(valid_extensions)]
    print(f"Found {len(files)} images to process.")

    for filename in files:
        img_path = os.path.join(input_folder, filename)

        with Image.open(img_path) as img:
            width, height = img.size
            min_dim = min(width, height)

            left = (width - min_dim) / 2
            top = (height - min_dim) / 2
            right = (width + min_dim) / 2
            bottom = (height + min_dim) / 2

            cropped_img = img.crop((left, top, right, bottom))

            resized_img = cropped_img.resize((256, 256), Image.Resampling.LANCZOS)

            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, f"{base_name}.png")

            resized_img.save(output_path, "PNG", dpi=(144, 144))

    print(f"Processing complete! All images saved to '{output_folder}'.")


if __name__ == "__main__":
    process_images()