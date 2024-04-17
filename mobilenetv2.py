import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Load pre-trained MobileNetV2 model
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Define transforms to preprocess the input image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess the input image
image_path = "/content/images.jpeg"
image = Image.open(image_path)
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

# Move the input and model to GPU for faster computation if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

# Perform prediction
with torch.no_grad():
    output = model(input_batch)

# Post-process the output (convert logits to probabilities)
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Get the predicted class index
predicted_class_idx = torch.argmax(probabilities).item()

print("Predicted class index:", predicted_class_idx)
print("Probability:", probabilities[predicted_class_idx].item())
