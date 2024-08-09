import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# 모델과 토크나이저를 로드합니다.
model_name = "Qwen/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print(model.config)


def truncate_emb_layer(layer, in_dim=None, out_dim=None):
    if in_dim is None:
        in_dim = layer.weight.data.size(0)
    if out_dim is None:
        out_dim = layer.weight.data.size(1)
    layer.weight.data = layer.weight.data[:in_dim, :out_dim].contiguous()
    layer.embedding_dim = out_dim


def truncate_layer_norm(layer, out_dim=None):
    if out_dim is None:
        out_dim = layer.weight.data.size(0)
    print(layer, out_dim)
    print(layer.weight.data.shape)
    layer.weight.data = layer.weight.data[:out_dim].contiguous()
    print(layer.weight.data.shape)
    print("-" * 80)
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data = layer.bias.data[:out_dim].contiguous()
    layer.normalized_shape = (out_dim,)


def truncate_linear_layer(layer, in_dim=None, out_dim=None):
    if in_dim is None:
        in_dim = layer.weight.data.size(1)
    if out_dim is None:
        out_dim = layer.weight.data.size(0)
    # print(layer, in_dim, out_dim)
    # print(layer.weight.data.shape)
    layer.weight.data = layer.weight.data[:out_dim, :in_dim].contiguous()
    # print(layer.weight.data.shape)
    # print("-" * 80)
    if not isinstance(layer, nn.Embedding) and layer.bias is not None:
        layer.bias.data = layer.bias.data[:out_dim].contiguous()
    layer.in_features = in_dim
    layer.out_features = out_dim


# 블록을 분할하는 함수입니다.
def quantize(model, k, hr=0.5):
    new_hidden_size = int(hr * model.config.hidden_size)
    new_intermediate_size = int(hr * model.config.intermediate_size)
    num_attention_heads = int(hr * model.config.num_attention_heads)
    num_key_value_heads = int(hr * model.config.num_key_value_heads)
    model.config.update(
        {
            "hidden_size": new_hidden_size,
            "intermediate_size": new_intermediate_size,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
        }
    )

    truncate_emb_layer(
        model.model.embed_tokens,
        in_dim=None,
        out_dim=new_hidden_size,
    )
    truncate_linear_layer(
        model.lm_head,
        in_dim=new_hidden_size,
        out_dim=None,
    )
    truncate_layer_norm(
        model.model.norm,
        new_hidden_size,
    )
    # 모델의 블록들을 가져옵니다.
    blocks = model.model.layers

    # 각 분할된 블록의 첫 번째 블록만 남깁니다.
    new_blocks = []
    for i in range(len(blocks) - 1, -1, -k):
        new_blocks.append(
            truncate_weights(
                model,
                blocks[i],
                new_hidden_size,
                new_intermediate_size,
            )
        )
    # 새롭게 만들어진 블록들로 모델을 업데이트합니다.
    model.model.layers = torch.nn.ModuleList(reversed(new_blocks))
    model.config.update({"num_hidden_layers": len(new_blocks)})
    return model


def truncate_weights(
    model,
    layer,
    new_hidden_size,
    new_intermediate_size,
):
    truncate_linear_layer(layer.self_attn.q_proj, new_hidden_size, new_hidden_size)
    kv_dim = int(
        new_hidden_size
        * model.config.num_key_value_heads
        / model.config.num_attention_heads
    )
    truncate_linear_layer(layer.self_attn.k_proj, new_hidden_size, kv_dim)
    truncate_linear_layer(layer.self_attn.v_proj, new_hidden_size, kv_dim)
    truncate_linear_layer(layer.self_attn.o_proj, new_hidden_size, new_hidden_size)
    truncate_linear_layer(layer.mlp.gate_proj, new_hidden_size, new_intermediate_size)
    truncate_linear_layer(layer.mlp.up_proj, new_hidden_size, new_intermediate_size)
    truncate_linear_layer(layer.mlp.down_proj, new_intermediate_size, new_hidden_size)
    truncate_layer_norm(layer.input_layernorm, new_hidden_size)
    truncate_layer_norm(layer.post_attention_layernorm, new_hidden_size)

    return layer


# 전체 파라미터 수를 계산하는 함수입니다.
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(model)
# 예시로 k=3으로 분할한 모델을 사용합니다.
k = 3
hd_ratio = 0.5

modified_model = quantize(model, k, hd_ratio)
print(modified_model)

# 전체 파라미터 수를 계산하고 출력합니다.
total_params = count_parameters(modified_model)
print(f"Total parameters: {total_params / 1e6:.2f} Million")

# 모델을 수정된 형태로 저장합니다.
model_suffix = model_name.split("/")[-1]
output_model_path = (
    f"/raid/channel/dobby/{model_suffix}-L{k}H{hd_ratio}Q{int(total_params / 1e6)}M"
)
modified_model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)

print(modified_model.config)
print(f"Modified model saved to {output_model_path}")

# 모델을 다시 로드할 때 변경된 구성 요소를 명시적으로 처리합니다.
loaded_model = AutoModelForCausalLM.from_pretrained(output_model_path)

# 다시 파라미터 수를 계산하여 확인합니다.
loaded_total_params = count_parameters(loaded_model)
print(f"Total parameters: {total_params / 1e6:.2f} Million")
