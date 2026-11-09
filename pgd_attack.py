import torch
import torchvision.models as models

# Select the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pretrained image classification model
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights).to(device)
model.eval()

print("Model loaded successfully!")
print("Device:", device)
print("Number of parameters:", sum(p.numel() for p in model.parameters()))
import urllib.request
from PIL import Image

# Download a sample image
image_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
image_path = "dog.jpg"

urllib.request.urlretrieve(image_url, image_path)

print("Image downloaded successfully!")
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# Load and prepare the image
image = Image.open(image_path).convert("RGB")
image = image.resize((224, 224))

to_tensor = T.ToTensor()
x = to_tensor(image).unsqueeze(0).to(device)

# Normalization required by the pretrained model
mean = torch.tensor(
    [0.485, 0.456, 0.406], device=device
).view(1, 3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225], device=device
).view(1, 3, 1, 1)

def classify(z):
    return model((z - mean) / std)

# Original prediction
with torch.no_grad():
    original_logits = classify(x)
    original_class = original_logits.argmax(dim=1).item()

categories = weights.meta["categories"]

print("\nOriginal prediction:", categories[original_class])

# PGD parameters
epsilon = 8 / 255
alpha = 2 / 255
steps = 10

criterion = nn.CrossEntropyLoss()
y = torch.tensor([original_class], device=device)

# Start with a small random perturbation
x_adv = x + torch.empty_like(x).uniform_(-epsilon, epsilon)
x_adv = torch.clamp(x_adv, 0, 1)

# Projected Gradient Descent attack
for step in range(steps):
    x_adv = x_adv.detach().requires_grad_(True)

    logits = classify(x_adv)
    loss = criterion(logits, y)

    gradient = torch.autograd.grad(loss, x_adv)[0]

    # Increase the loss
    x_adv = x_adv.detach() + alpha * gradient.sign()

    # Project onto the allowed perturbation set
    perturbation = torch.clamp(x_adv - x, -epsilon, epsilon)
    x_adv = torch.clamp(x + perturbation, 0, 1).detach()

# Evaluate the adversarial image
with torch.no_grad():
    adversarial_logits = classify(x_adv)
    adversarial_class = adversarial_logits.argmax(dim=1).item()
    final_loss = criterion(adversarial_logits, y).item()

actual_perturbation = (x_adv - x).abs().max().item()

print("Adversarial prediction:", categories[adversarial_class])
print("Final loss:", final_loss)
print("Maximum perturbation:", actual_perturbation)
print("Attack successful:", adversarial_class != original_class)

# Save the adversarial image
T.ToPILImage()(x_adv.squeeze(0).cpu()).save("adversarial_image.png")

print("Adversarial image saved successfully!")
print("Epsilon:", epsilon)

print(
    "Perturbation constraint satisfied:",
    actual_perturbation <= epsilon + 1e-6
)

print(
    "Pixel range constraint satisfied:",
    x_adv.min().item() >= 0
    and x_adv.max().item() <= 1
)
