"""Görüntü başlıklandırma için yardımcı fonksiyonlar."""

import math
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoModel, AutoTokenizer

# torchvision.transforms aliası
# pylint: disable=invalid-name
Transforms = T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int) -> Transforms.Compose:
    """Verilen giriş boyutuna göre bir torchvision transform oluşturur.

    Args:
        input_size: Görüntünün dönüştürüleceği boyut (kare).

    Returns:
        Bir torchvision.transforms.Compose nesnesi.
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
    """Verilen en boy oranına en yakın hedef en boy oranını bulur.

    Args:
        aspect_ratio: Orijinal görüntünün en boy oranı.
        target_ratios: Hedef en boy oranları kümesi (genişlik_oranı, yükseklik_oranı).
        width: Orijinal görüntü genişliği.
        height: Orijinal görüntü yüksekliği.
        image_size: Tek bir yamanın boyutu.

    Returns:
        En iyi eşleşen (genişlik_oranı, yükseklik_oranı) demeti.
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
            # Eşitlik durumunda, orijinal koddaki bu koşul vardı ancak içi boştu.
            # Bu, genellikle ek bir sezgisel yöntemin (örn. daha büyük yama sayısı tercihi)
            # uygulanabileceği anlamına gelir. Şimdilik, ilk bulunan en iyi oranı koruyoruz.
            # Orijinal kodda bu if bloğunun içi boştu:
            # `if area > 0.5 * image_size * image_size * ratio_w * ratio_h:`
            # Bu nedenle burada bir `pass` kullanıyoruz.
            if area > 0.5 * image_size * image_size * ratio_w * ratio_h:
                pass  # Orijinal kodda bu blok boştu
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> List[Image.Image]:
    """Görüntüyü dinamik olarak en boy oranına göre yeniden boyutlandırır ve böler.

    Args:
        image: İşlenecek PIL.Image nesnesi.
        min_num: Minimum yama sayısı.
        max_num: Maksimum yama sayısı.
        image_size: Her bir yamanın hedef boyutu.
        use_thumbnail: Küçük resim eklenip eklenmeyeceği.

    Returns:
        İşlenmiş PIL.Image nesnelerinin bir listesi.
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
    )  # i*j kontrolü eklendi

    # (i,j) ve (j,i) varyasyonlarını ekle
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
    """Bir görüntü dosyasını yükler, işler ve tensöre dönüştürür.

    Args:
        image_file: Görüntü dosyasının yolu.
        input_size: Görüntü yamalarının boyutu.
        max_num: Maksimum yama sayısı.

    Returns:
        İşlenmiş görüntü yamalarını içeren bir torch.Tensor.
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
    """InternVL modeli için özel bir aygıt haritası oluşturur.

    Bu fonksiyon, InternVL model mimarisine özgüdür.

    Args:
        model_name_or_path: Model adı veya yolu.
        world_size: Kullanılacak GPU sayısı. None ise otomatik olarak algılanır.

    Returns:
        Katman adlarından aygıt kimliklerine bir eşleme sözlüğü.
    """
    if world_size is None:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    if world_size == 1:  # Tek GPU veya CPU durumu
        return {"": "cuda:0" if torch.cuda.is_available() else "cpu"}

    device_map: Dict[str, int] = {}
    try:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        if not hasattr(config, "llm_config") or not hasattr(
            config.llm_config, "num_hidden_layers"
        ):
            # print("Uyarı: Model yapılandırması beklenen InternVL formatında değil. 'auto' device_map kullanılacak.")
            return "auto"  # type: ignore[return-value]

        num_layers = config.llm_config.num_hidden_layers
        # İlk GPU ViT için kullanılacağından, yarım GPU olarak kabul edilir.
        num_layers_per_gpu_float = num_layers / (world_size - 0.5)

        num_layers_on_first_gpu = math.ceil(num_layers_per_gpu_float * 0.5)
        num_layers_on_other_gpus = math.ceil(num_layers_per_gpu_float)

        layer_cnt = 0
        # vision_model ve mlp1 ilk GPU'ya (0)
        device_map["vision_model"] = 0
        device_map["mlp1"] = 0

        # Dil modelinin gömme katmanları ilk GPU'ya
        # Bu katman adları modele göre değişebilir, genel bir varsayım yapılıyor.
        # Gerçek InternVL modelindeki kesin adlara göre ayarlanmalıdır.
        # Örnek adlar: language_model.model.tok_embeddings, language_model.model.embed_tokens
        # Bu adlar AutoConfig'den dinamik olarak alınabilirse daha iyi olur.
        # Şimdilik, bu katmanların var olduğunu ve ilk GPU'ya yerleştirileceğini varsayıyoruz.
        # Eğer modelde bu katmanlar yoksa veya adları farklıysa, device_map'te görünmeyeceklerdir.
        # Bu genellikle sorun yaratmaz, çünkü from_pretrained kalanları otomatik olarak yerleştirir.

        # LLM katmanlarının dağıtımı
        current_gpu_idx = 0
        layers_on_current_gpu = (
            num_layers_on_first_gpu
            if current_gpu_idx == 0
            else num_layers_on_other_gpus
        )

        for i in range(num_layers):
            if layers_on_current_gpu == 0:
                current_gpu_idx += 1
                if current_gpu_idx >= world_size:  # GPU'lar biterse sonuncuya ata
                    current_gpu_idx = world_size - 1
                layers_on_current_gpu = num_layers_on_other_gpus

            device_map[f"language_model.model.layers.{i}"] = current_gpu_idx
            layers_on_current_gpu -= 1

        # Kalan önemli katmanlar (genellikle son GPU'ya veya ilk GPU'ya)
        # Bu da modele özgüdür. InternVL'nin yapısına göre ayarlanmalıdır.
        # Örnek: language_model.model.norm, language_model.lm_head
        # Genellikle, belirtilmeyen katmanlar `device_map="auto"` mantığıyla yerleştirilir
        # veya `device_map` sözlüğünde olmayanlar ilk cihaza (genellikle 0) atanır.
        # Bu nedenle, tüm katmanları burada belirtmek zorunlu olmayabilir.
        # Ancak, InternVL'nin orijinal `split_model` fonksiyonu bazılarını belirtir.

        # Örnek olarak, bazı son katmanları son kullanılan GPU'ya veya ilk GPU'ya atayalım.
        final_gpu_idx = current_gpu_idx  # Son katmanların yerleştirildiği GPU

        # Bu katmanların varlığı ve adları modele göre değişir.
        # `language_model.model.tok_embeddings` gibi katmanlar genellikle başa konur.
        # `language_model.output` veya `language_model.lm_head` gibi katmanlar sona konur.
        # InternVL'nin orijinal `split_model` fonksiyonundaki mantığı yansıtmaya çalışalım:
        device_map["language_model.model.tok_embeddings"] = 0
        device_map["language_model.model.embed_tokens"] = 0  # Bazen bu kullanılır
        device_map["language_model.output"] = final_gpu_idx  # Veya lm_head
        device_map["language_model.lm_head"] = final_gpu_idx
        device_map["language_model.model.norm"] = final_gpu_idx
        # Rotary embeddings genellikle her katmanda yeniden oluşturulur veya paylaşılır,
        # bu nedenle doğrudan bir cihaza atanmaz.
        # device_map[f"language_model.model.layers.{num_layers - 1}"] = final_gpu_idx # Zaten döngüde atanmış olmalı

    except Exception as e:
        # print(f"InternVL için özel aygıt haritası oluşturulurken hata: {e}. 'auto' kullanılacak.")
        return "auto"  # type: ignore[return-value]

    # print(f"Oluşturulan InternVL aygıt haritası: {device_map}")
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
    """Verilen parametrelerle bir modeli ve tokenizer'ı yükler.

    Args:
        model_name_or_path: Model adı veya Hugging Face Hub yolu.
        torch_dtype_str: Kullanılacak torch veri türü (örn. "bfloat16", "float16", "float32").
        load_in_8bit: Modelin 8-bit olarak yüklenip yüklenmeyeceği.
        use_flash_attn: Flash Attention kullanılıp kullanılmayacağı.
        device_map: Aygıt eşleme stratejisi ("auto", "balanced", "none") veya özel bir sözlük.
        trust_remote_code: Uzak kodun güvenilip güvenilmeyeceği.
        low_cpu_mem_usage: Düşük CPU bellek kullanımı etkinleştirilip etkinleştirilmeyeceği.

    Returns:
        (model, tokenizer) içeren bir demet.

    Raises:
        ValueError: Geçersiz torch_dtype_str sağlanırsa.
    """
    if hasattr(torch, torch_dtype_str):
        dtype = getattr(torch, torch_dtype_str)
    else:
        raise ValueError(
            f"Geçersiz torch_dtype_str: {torch_dtype_str}. "
            f"torch.{torch_dtype_str} bulunamadı."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        use_fast=False,  # InternVL orijinalinde use_fast=False idi
    )

    model_kwargs = {
        "torch_dtype": dtype,
        "load_in_8bit": load_in_8bit,
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
    }
    # use_flash_attn sadece bazı modeller ve transformers versiyonları tarafından desteklenir.
    # Desteklenmiyorsa hata vermemesi için koşullu ekleme yapalım.
    if use_flash_attn:
        # Flash Attention 2 için `attn_implementation="flash_attention_2"` kullanılır.
        # `use_flash_attn` eski bir parametre olabilir.
        # Yeni HF versiyonları `attn_implementation` bekler.
        # Şimdilik, eğer model config'i destekliyorsa ekleyelim.
        # Ya da doğrudan `attn_implementation` olarak geçelim.
        # InternVL `use_flash_attn=True` kullanıyordu, bu muhtemelen modelin kendi argümanı.
        model_kwargs["use_flash_attn"] = True  # InternVL'nin kullandığı gibi

    model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)  # type: ignore[arg-type]

    model.eval()  # Modeli değerlendirme moduna al

    return model, tokenizer
