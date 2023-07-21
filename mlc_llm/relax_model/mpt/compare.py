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

def advanced_compare(lft, rht, atol=1e-5, rtol=1e-5):
  if len(lft.shape) > 1:
    lft = lft.flatten()
  if len(rht.shape) > 1:
    lft = rht.flatten()
  numel = lft.shape[0]
  assert numel == rht.shape[0]
  for i in range(numel):
    if np.abs(lft[i]-rht[i]) > atol + rtol*np.abs(rht[i]):
      print("Elements with index", i, " are not the same left:", lft[i], " right:", rht[i])

def main():
  check_num = 10
  rtol=1e-1
  atol=1e-8
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
  print("ORIG INPUT:", orig_np_input[:check_num])
  print("RELAX INPUT:", np_input[:check_num])
  # np.testing.assert_allclose(orig_np_input, np_input, rtol=rtol, atol=atol, verbose=True)
  advanced_compare(orig_np_input, np_input, rtol=rtol, atol=atol)

  print("Compare weights")
  orig_np_line = orig_np_weight[0,:]
  print("ORIG WEIGHT:", orig_np_line[:check_num])
  print("RELAX WEIGHT:", np_weight[:check_num])
  np.testing.assert_allclose(orig_np_line, np_weight, rtol=rtol, atol=atol, verbose=True)

if __name__ == "__main__":
  main()
