"""Utility functions for algorithm-specific parameter handling."""

from typing import Dict, Any


def reconstruct_full_critic_params(critic_params: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Reconstruct full critic parameters from the dict structure stored in info.
    
    The critic_params dict has structure {0: params_dict_0, 1: params_dict_1, ...}
    where each params_dict_i contains the parameters for critic i.
    
    This function reconstructs the full parameter structure needed by critic.apply():
    - For CRL: {"sa_encoder_0": ..., "g_encoder_0": ..., "sa_encoder_1": ..., "g_encoder_1": ...}
    - For SAC: {"critic_0_hidden_0": ..., "critic_0_output": ..., "critic_1_hidden_0": ..., ...}
    
    Args:
        critic_params: Dict mapping critic index to its parameter dict
        
    Returns:
        Full critic parameters dict in the format expected by critic.apply()
    """
    if not critic_params:
        return {}
    
    # Check the structure of the first critic's params to determine format
    first_critic_params = critic_params[0]
    is_crl_format = "sa_encoder" in first_critic_params or "g_encoder" in first_critic_params
    
    full_critic_params = {}
    for critic_idx, critic_i_params in critic_params.items():
        if is_crl_format:
            # CRL format: {"sa_encoder": ..., "g_encoder": ...}
            # Reconstruct as {"sa_encoder_{i}": ..., "g_encoder_{i}": ...}
            if "sa_encoder" in critic_i_params:
                full_critic_params[f"sa_encoder_{critic_idx}"] = critic_i_params["sa_encoder"]
            if "g_encoder" in critic_i_params:
                full_critic_params[f"g_encoder_{critic_idx}"] = critic_i_params["g_encoder"]
        else:
            # SAC format: {"hidden_0": ..., "hidden_1": ..., "output": ...}
            # Reconstruct as {"critic_{i}_hidden_0": ..., "critic_{i}_output": ...}
            for layer_name, layer_params in critic_i_params.items():
                full_critic_params[f"critic_{critic_idx}_{layer_name}"] = layer_params
    
    return full_critic_params
