"""Utility functions for image captioning."""

import math
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoModel, AutoTokenizer

# torchvision.transforms alias
# pylint: disable=invalid-name
Transforms = T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int) -> Transforms.Compose:
    """Builds a torchvision transform for the given input size.

    Args:
        input_size: Target size for the image transformation (square).

    Returns:
        A torchvision.transforms.Compose object.
    """
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    transform = Transforms.Compose(
        [
            Transforms.Lambda(
                lambda img: img.convert("RGB") if img.mode != "RGB" else img
            ),
            Transforms.Resize(
                (input_size, input_size), interpolation=InterpolationMode.BICUBIC
            ),
            Transforms.ToTensor(),
            Transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transform


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: Set[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    """Finds the closest target aspect ratio to the given aspect ratio.

    Args:
        aspect_ratio: Original image aspect ratio.
        target_ratios: Set of target aspect ratios (width_ratio, height_ratio).
        width: Original image width.
        height: Original image height.
        image_size: Size of a single patch.

    Returns:
        Best matching (width_ratio, height_ratio) tuple.
    """
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio_w, ratio_h in target_ratios:
        target_aspect_ratio = ratio_w / ratio_h
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)

        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = (ratio_w, ratio_h)
        elif ratio_diff == best_ratio_diff:
            # In case of equality, this condition existed in the original code but was empty.
            # This usually means additional heuristic methods (e.g., preference for larger patch count)
            # could be applied. For now, we keep the first best ratio found.
            # In the original code, this if block was empty:
            # `if area > 0.5 * image_size * image_size * ratio_w * ratio_h:`
            # Therefore, we use a `pass` here.
            if area > 0.5 * image_size * image_size * ratio_w * ratio_h:
                pass  # Original code had this block empty
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> List[Image.Image]:
    """Dynamically resizes and splits the image based on aspect ratio.

    Args:
        image: PIL.Image object to be processed.
        min_num: Minimum number of patches.
        max_num: Maximum number of patches.
        image_size: Target size for each patch.
        use_thumbnail: Whether to add a thumbnail.

    Returns:
        A list of processed PIL.Image objects.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n_patches in range(min_num, max_num + 1)
        for i in range(1, int(math.sqrt(n_patches)) + 1)
        if n_patches % i == 0
        for j in [n_patches // i]
        if i * j <= max_num and i * j >= min_num
    )  # i*j check added

    # Add (i,j) and (j,i) variations
    additional_ratios = set()
    for r_w, r_h in target_ratios:
        additional_ratios.add((r_w, r_h))
        additional_ratios.add((r_h, r_w))
    target_ratios = sorted(list(additional_ratios), key=lambda x: x[0] * x[1])

    target_aspect_config = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_config[0]
    target_height = image_size * target_aspect_config[1]
    num_blocks = target_aspect_config[0] * target_aspect_config[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(num_blocks):
        box = (
            (i % target_aspect_config[0]) * image_size,
            (i // target_aspect_config[0]) * image_size,
            ((i % target_aspect_config[0]) + 1) * image_size,
            ((i // target_aspect_config[0]) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == num_blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(
    image_file: str, input_size: int = 448, max_num: int = 12
) -> torch.Tensor:
    """Loads an image file, processes it, and converts it to tensor.

    Args:
        image_file: Path to the image file.
        input_size: Size of image patches.
        max_num: Maximum number of patches.

    Returns:
        A torch.Tensor containing processed image patches.
    """
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(img) for img in images]
    pixel_values_stacked = torch.stack(pixel_values)
    return pixel_values_stacked


def split_model_internvl(
    model_name_or_path: str, world_size: Optional[int] = None
) -> Dict[str, int]:
    """Creates a custom device map for InternVL model.

    This function is specific to the InternVL model architecture.

    Args:
        model_name_or_path: Model name or path.
        world_size: Number of GPUs to use. If None, automatically detected.

    Returns:
        A mapping dictionary from layer names to device IDs.
    """
    if world_size is None:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    if world_size == 1:  # Single GPU or CPU case
        return {"": "cuda:0" if torch.cuda.is_available() else "cpu"}

    device_map: Dict[str, int] = {}
    try:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        if not hasattr(config, "llm_config") or not hasattr(
            config.llm_config, "num_hidden_layers"
        ):
            # print("Warning: Model configuration is not in expected InternVL format. 'auto' device_map will be used.")
            return "auto"  # type: ignore[return-value]

        num_layers = config.llm_config.num_hidden_layers
        # Since the first GPU will be used for ViT, it's considered as half GPU.
        num_layers_per_gpu_float = num_layers / (world_size - 0.5)

        num_layers_on_first_gpu = math.ceil(num_layers_per_gpu_float * 0.5)
        num_layers_on_other_gpus = math.ceil(num_layers_per_gpu_float)

        layer_cnt = 0
        # vision_model and mlp1 to first GPU (0)
        device_map["vision_model"] = 0
        device_map["mlp1"] = 0

        # Language model embedding layers to first GPU
        # These layer names may vary by model, making a general assumption.
        # Should be adjusted according to exact names in actual InternVL model.
        # Example names: language_model.model.tok_embeddings, language_model.model.embed_tokens
        # It would be better if these names could be dynamically obtained from AutoConfig.
        # For now, we assume these layers exist and will be placed on the first GPU.
        # If these layers don't exist in the model or have different names, they won't appear in device_map.
        # This usually doesn't cause problems, as from_pretrained automatically places the rest.

        # Distribution of LLM layers
        current_gpu_idx = 0
        layers_on_current_gpu = (
            num_layers_on_first_gpu
            if current_gpu_idx == 0
            else num_layers_on_other_gpus
        )

        for i in range(num_layers):
            if layers_on_current_gpu == 0:
                current_gpu_idx += 1
                if current_gpu_idx >= world_size:  # If GPUs run out, assign to last one
                    current_gpu_idx = world_size - 1
                layers_on_current_gpu = num_layers_on_other_gpus

            device_map[f"language_model.model.layers.{i}"] = current_gpu_idx
            layers_on_current_gpu -= 1

        # Remaining important layers (usually to last GPU or first GPU)
        # This is also model-specific. Should be adjusted according to InternVL's structure.
        # Example: language_model.model.norm, language_model.lm_head
        # Usually, unspecified layers are placed with `device_map="auto"` logic
        # or those not in `device_map` dictionary are assigned to first device (usually 0).
        # Therefore, it may not be mandatory to specify all layers here.
        # However, InternVL's original `split_model` function specifies some of them.

        # As an example, let's assign some final layers to the last used GPU or first GPU.
        final_gpu_idx = current_gpu_idx  # GPU where final layers are placed

        # The existence and names of these layers vary by model.
        # Layers like `language_model.model.tok_embeddings` are usually placed at the beginning.
        # Layers like `language_model.output` or `language_model.lm_head` are placed at the end.
        # Let's try to reflect the logic from InternVL's original `split_model` function:
        device_map["language_model.model.tok_embeddings"] = 0
        device_map["language_model.model.embed_tokens"] = 0  # Sometimes this is used
        device_map["language_model.output"] = final_gpu_idx  # Or lm_head
        device_map["language_model.lm_head"] = final_gpu_idx
        device_map["language_model.model.norm"] = final_gpu_idx
        # Rotary embeddings are usually recreated or shared in each layer,
        # so they are not directly assigned to a device.
        # device_map[f"language_model.model.layers.{num_layers - 1}"] = final_gpu_idx # Should already be assigned in the loop

    except Exception as e:
        # print(f"Error creating custom device map for InternVL: {e}. 'auto' will be used.")
        return "auto"  # type: ignore[return-value]

    # print(f"Generated InternVL device map: {device_map}")
    return device_map


def load_model_and_tokenizer(
    model_name_or_path: str,
    torch_dtype_str: str = "bfloat16",
    load_in_8bit: bool = False,
    use_flash_attn: bool = True,
    device_map: Union[str, Dict[str, Any]] = "auto",
    trust_remote_code: bool = True,
    low_cpu_mem_usage: bool = True,
) -> Tuple[AutoModel, AutoTokenizer]:
    """Loads a model and tokenizer with the given parameters.

    Args:
        model_name_or_path: Model name or Hugging Face Hub path.
        torch_dtype_str: Torch data type to use (e.g., "bfloat16", "float16", "float32").
        load_in_8bit: Whether to load the model in 8-bit.
        use_flash_attn: Whether to use Flash Attention.
        device_map: Device mapping strategy ("auto", "balanced", "none") or a custom dictionary.
        trust_remote_code: Whether to trust remote code.
        low_cpu_mem_usage: Whether to enable low CPU memory usage.

    Returns:
        A tuple containing (model, tokenizer).

    Raises:
        ValueError: If an invalid torch_dtype_str is provided.
    """
    if hasattr(torch, torch_dtype_str):
        dtype = getattr(torch, torch_dtype_str)
    else:
        raise ValueError(
            f"Invalid torch_dtype_str: {torch_dtype_str}. "
            f"torch.{torch_dtype_str} not found."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        use_fast=False,  # use_fast=False in original InternVL
    )

    model_kwargs = {
        "torch_dtype": dtype,
        "load_in_8bit": load_in_8bit,
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
    }
    # use_flash_attn is only supported by some models and transformers versions.
    # Let's add it conditionally so it doesn't error if not supported.
    if use_flash_attn:
        # For Flash Attention 2, `attn_implementation="flash_attention_2"` is used.
        # `use_flash_attn` might be an older parameter.
        # New HF versions expect `attn_implementation`.
        # For now, if the model config supports it, let's add it.
        # Or pass it directly as `attn_implementation`.
        # InternVL used `use_flash_attn=True`, this is probably the model's own argument.
        model_kwargs["use_flash_attn"] = True  # As used by InternVL

    model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)  # type: ignore[arg-type]

    model.eval()  # Set model to evaluation mode

    return model, tokenizer
