import torch

def remove_module_from_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # Remove 'module.' from the key
        new_state_dict[new_key] = value
    return new_state_dict

path = "/home/ziyao/deblurvit/experiments/2024-09-10_19-24-39_train_depthstripformer/models/model_best.pth"
ckpt = torch.load(path,map_location='cpu')
ckpt['model_state_dict'] = remove_module_from_state_dict(ckpt['model_state_dict'])
torch.save(ckpt,path)
