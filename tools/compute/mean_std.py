# Author: Yao
# CreatTime: 2025/10/8
# FileName: mean_std
# Description: simple introduction of the code
import torch


def compute_mean_std(tensor):
    """
    Compute mean and std of a tensor.

    Args:
        tensor: input tensor

    Returns:
        mean: mean value
        std: standard deviation
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, dtype=torch.float32)

    mean = torch.mean(tensor)

    std = torch.std(tensor)

    return mean, std


# Examples
if __name__ == "__main__":
    # 1. Random data
    # DIOR
    # map_b = torch.tensor([64.39,64.37,64.33,64.45,64.42])
    # map_a = 4 / 20 * map_n + 16 / 20 * map_b
    # HM = 2 * map_n*map_b / (map_n+map_b)
    # DOTA
    map_n = torch.tensor([24.3688, 23.2896, 23.7383, 23.4520, 23.9087])
    map_b = torch.tensor([62.03, 62.05, 61.97, 61.97, 61.98])
    map_a = 4/15 * map_n + 11/15 * map_b
    hm = 2*map_n*map_b/(map_n+map_b)
    mean, std = compute_mean_std(hm)
    print(f"custom data - mean: {mean:.4f}, std: {std:.4f}")

    # map_b = torch.tensor([64.4, 75.5, 68.7,  63.5])
    # # map_a_DOTA = 4/15 * map_n + 11/15 * map_b
    # map_a_DIOR = 4 / 20 * map_n + 16 / 20 * map_b
    # hm = 2 * map_n*map_b / (map_n+map_b)
    # Rel_AP = (map_b - map_n) / map_b
