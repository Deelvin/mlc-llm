from pathlib import Path

import torch
import numpy as np

# std::ofstream fs("tensor.bin", std::ios::out | std::ios::binary | std::ios::app);
# fs.write(reinterpret_cast<const char*>(&tensor), sizeof tensor);
# fs.close();

def save_torch_tensor(t: torch.tensor, path=Path("./orig_input.pt")):
  torch.save(t, path)

def load_torch_tensor(path=Path("./orig_input.pt")):
  return torch.load(path)

def main():
  # Load data from Relax model
  np_input = np.fromfile(Path("./relax_input.bin"), dtype="float32")
  np_weight = np.fromfile(Path("./relax_weight.bin"), dtype="float32")
  print("RELAX INPUT TYPE:", np_input.dtype, "SHAPE:", np_input.shape)
  print("RELAX WEIGHT TYPE:", np_weight.dtype, "SHAPE:", np_weight.shape)

  # Load data from original model
  orig_input = load_torch_tensor()
  orig_weight = load_torch_tensor(Path("./orig_weight.pt"))

  orig_np_input = orig_input.numpy()
  orig_np_weight = orig_weight.numpy()
  print("ORIG INPUT TYPE:", orig_np_input.dtype, "SHAPE:", orig_np_input.shape)
  print("ORIG WEIGHT TYPE:", orig_np_weight.dtype, "SHAPE:", orig_np_weight.shape)

  print("Compare inputs")
  np.allclose(orig_np_input, np_input, atol=1e-3)
  print("Compare weights")
  np.allclose(orig_np_weight[0,:], np_weight, atol=1e-3)

if __name__ == "__main__":
  main()
