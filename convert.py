import os
import torch
import coremltools as ct
import importlib.util

# parse input shape
shape = tuple(map(int, os.environ["MODEL_INPUT"].split(",")))
model_name = os.environ["MODEL_NAME"]

# dynamic import
spec = importlib.util.spec_from_file_location("model_module", "model.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

model = m.ModelStructure()
model.load_state_dict(torch.load("best.pt", map_location="cpu"))
model.eval()

example_input = torch.randn(*shape)
traced = torch.jit.trace(model, example_input)

mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="input", shape=shape)],
)

mlmodel.save(f"{model_name}.mlpackage")
