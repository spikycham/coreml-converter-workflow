import torch
import coremltools as ct
from torch import nn
from predict import ModelRegressor


# ==== load model ====
model = ModelRegressor()
state_dict = torch.load("best.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# ==== trace ====
example_input = torch.randn(1, 3, 300, 300)
traced_model = torch.jit.trace(model, example_input)

# ==== convert to Core ML ====
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(
            name="image",
            shape=example_input.shape,
        )
    ],
    compute_units=ct.ComputeUnit.ALL,
)

# ==== save ====
mlmodel.save("ios.mlpackage")

# print("✅ Converted to ios.mlpackage")
