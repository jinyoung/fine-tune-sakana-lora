"""Test Doc-to-LoRA inference on CPU (Mac)."""
import torch
import sys

# Patch for Mac: disable flash attention references
from ctx_to_lora.modeling import hypernet
sys.modules["ctx_to_lora.modeling_utils"] = hypernet

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

# Use Qwen 4B checkpoint (not gated)
checkpoint_path = "trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"
print(f"Loading checkpoint: {checkpoint_path}")

state_dict = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
print("State dict loaded. Building model...")

# Patch: disable quantization for ctx_encoder on Mac
ctx_encoder_args = state_dict["ctx_encoder_args"]
ctx_encoder_args.quantize_ctx_encoder = False
print(f"Patched ctx_encoder_args: quantize_ctx_encoder=False")

model = ModulatedPretrainedModel.from_state_dict(
    state_dict,
    train=False,
    use_flash_attn=False,  # No flash attention on Mac
    use_sequence_packing=False,
    base_model_kwargs={"device_map": "cpu", "torch_dtype": torch.float32},
)
model.eval()
print(f"Model loaded! Base model: {model.base_model.config.name_or_path}")

tokenizer = get_tokenizer(model.base_model.config.name_or_path)

# Prepare document and question
doc = open("data/sakana_wiki.txt", "r").read()
print(f"\nDocument:\n{doc[:200]}...\n")

chat = [{"role": "user", "content": "Tell me about Sakana AI."}]
chat_ids = tokenizer.apply_chat_template(
    chat,
    add_special_tokens=False,
    return_attention_mask=False,
    add_generation_prompt=True,
    return_tensors="pt",
)

print("Internalizing document...")
model.internalize(doc)
print("Document internalized! Generating response...")

with torch.inference_mode():
    outputs = model.generate(input_ids=chat_ids, max_new_tokens=128)
    response = tokenizer.decode(outputs[0][chat_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nResponse (with internalization):\n{response}")

# Reset and try without internalization
model.reset()
print("\n" + "="*60)
print("Now generating WITHOUT internalization...")
with torch.inference_mode():
    outputs = model.generate(input_ids=chat_ids, max_new_tokens=128)
    response = tokenizer.decode(outputs[0][chat_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nResponse (without internalization):\n{response}")
