
import importlib

node_list = ["image_node", "qwen_node", "RMBG_node", "DepthAnythingV2_node", "SeedVR2_node","openrouter_vision_node", "Fal_Auth_node", "openrouter_LLM_API_node", "seedream_image_node", "krea2_turbo_lora_node"]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module_name in node_list:
    module = importlib.import_module(f".nodes.{module_name}", package=__name__)
    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
    

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
