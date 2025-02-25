from PIL import Image
import os
# image_id c_gt c_target
img_dir = "D:\Transformation-Based-Adversarial-Attack-To-VLM\Flickr8k_Dataset\Flicker8k_Dataset"
annotation_file = "D:\Transformation-Based-Adversarial-Attack-To-VLM\Flickr8k_text\Flickr8k.token.txt"
with open(annotation_file, 'r') as f:
    lines = [line.strip().split("\t") for line in f.readlines()]
    image_ids = [line[0].split("#")[0] for line in lines]
    texts = [line[1].split("\t")[1] for line in lines]
    img = Image.open(os.path.join(img_dir, image_ids[0]))
    img.show()
    

