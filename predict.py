import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.efficientnet import efficientnet_b0, EfficientNet_B0_Weights
from torch import nn

# ==== 模型定义 ====
class ModelRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        self.cnn = efficientnet_b0(weights=weights)
        self.cnn.classifier = nn.Identity()
        self.feature_dim = self.cnn.features[-1][0].out_channels

        self.regressor = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, image):
        img_feat = self.cnn(image)
        img_feat = torch.flatten(img_feat, 1)
        mu = self.regressor(img_feat).squeeze(1)
        return mu

# ==== 图像预处理 ====
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==== 推理函数 ====
def predict(img_path, model_path="./efficientnetb0.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ModelRegressor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    try:
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_sqrt = model(image).item()
            pred = pred_sqrt ** 2
        return pred
    except Exception as e:
        return -1


# ==== 主程序入口 ====
if __name__ == "__main__":
    img = "./79.jpg"  # 可改为你想推理的图片目录

    # df = predict(img_paths, model_path="./efficientnetb0.pt")
    # df.to_csv("inference_result.csv", index=False)
    # print("✅ 推理完成，结果保存在 inference_result.csv")

    a = predict(img)
    print(a)