import torch.nn as nn
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    VGG16_Weights,
    densenet121,
    efficientnet_b0,
    resnet50,
    vgg16,
)


MODEL_NAMES = (
    "ButterflyC",
    "VGG16M",
    "ResNet50M",
    "DenseNet121M",
)


def normalize_model_name(model_name):
    if model_name in MODEL_NAMES:
        return model_name
    return "ButterflyC"


def build_torch_model(model_name, num_classes, *, pretrained=True):
    resolved_model_name = normalize_model_name(model_name)
    weights = None

    if resolved_model_name == "ButterflyC":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        head_parameters = list(model.classifier.parameters())
    elif resolved_model_name == "VGG16M":
        weights = VGG16_Weights.DEFAULT if pretrained else None
        model = vgg16(weights=weights)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        head_parameters = list(model.classifier.parameters())
    elif resolved_model_name == "ResNet50M":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        head_parameters = list(model.fc.parameters())
    else:
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        model = densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        head_parameters = list(model.classifier.parameters())

    return resolved_model_name, model, head_parameters


def freeze_backbone(model, head_parameters):
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in head_parameters:
        parameter.requires_grad = True


def unfreeze_model(model):
    for parameter in model.parameters():
        parameter.requires_grad = True


def count_trainable_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
