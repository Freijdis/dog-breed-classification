import os
import io
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import json


def model_fn(model_dir):
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 133)

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=torch.device('cpu')))

    model.eval()
    return model


def input_fn(request_body, content_type):
    if content_type not in ['image/jpeg', 'image/png']:
        raise ValueError(f"Unsupported content type: {content_type}")

    image = Image.open(io.BytesIO(request_body))

    transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transformation(image).unsqueeze(0)


def predict_fn(input_data, model):
    with torch.no_grad():
        output = model(input_data)
        _, predicted_class = torch.max(output, 1)
    return predicted_class.item() + 1


def output_fn(prediction, accept):
    if accept == "application/json":
        return json.dumps({"predicted_class": prediction}), "application/json"
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
