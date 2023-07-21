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
  np_input = np.fromfile(Path("./relax_input.bin"), dtype="float")
  np_weight = np.fromfile(Path("./relax_weight.bin"), dtype="float")
  
  # np_input = np_input.astype(float)
  # np_weight = np_weight.astype(float)
  
  # Load data from original model
  orig_input = load_torch_tensor()
  orig_weight = load_torch_tensor(Path("./orig_weight.pt"))
  
  orig_np_input = orig_input.astype(torch.float).numpy()
  orig_np_weight = orig_weight.astype(torch.float).numpy()
  
  print("Compare inputs")
  np.allclose(orig_np_input, np_input, atol=1e-3)
  print("Compare weights")
  np.allclose(orig_np_weight, np_weight, atol=1e-3)

if __name__ == "__main__":
  main()
