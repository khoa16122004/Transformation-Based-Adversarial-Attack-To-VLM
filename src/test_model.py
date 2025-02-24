import torch
import clip
from PIL import Image
import open_clip
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained='laion2B-s32B-b79K')
tokenizer = open_clip.get_tokenizer("ViT-H-14")
model.eval().cuda()
print("Khoa")
img = cv2.resize(cv2.imread('output.jpg'), (224, 224))
image = preprocess(Image.fromarray(img)).cuda().unsqueeze(0) # 
text = tokenizer(["A proud sea lion basking in the sun on the rocks.", 
                  "who are you?", "A dog runing in the grass."]).to(device)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    print(text_probs.sum())
print("Label probs:", text_probs * 100)  # prints: [[1., 0., 0.]]