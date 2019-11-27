from torchvision.models import resnet18

model = resnet18()
for name, param in model.named_parameters():
    print(name)

print(model.conv1)