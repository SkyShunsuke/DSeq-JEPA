import torch

def concat_results_dict(dict_A, dict_B):
    """Concatenate two result dictionaries along the first dimension.
    Args:
        dict_A: dict, key -> tensor of shape (N1, ...)
        dict_B: dict, key -> tensor of shape (N2, ...)
    Returns:
        result_dict: dict, key -> tensor of shape (N1 + N2, ...)
    """
    dict_list = [dict_A, dict_B]
    result_dict = {}
    for d in dict_list:
        for k in d:
            if k not in result_dict:
                result_dict[k] = []
            result_dict[k].append(d[k])
    for k in result_dict:
        result_dict[k] = torch.cat(result_dict[k], dim=0)
    return result_dict