import zipfile

def unzip_file(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# Ví dụ:
unzip_file("Flickr8k_Dataset.zip", "imgs")