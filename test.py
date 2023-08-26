import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
    print(torch.backends.mps.is_built())
    print(torch.__version__)
    print(torch.mps.empty_cache)
else:
    print ("MPS device not found.")