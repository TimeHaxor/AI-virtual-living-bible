
import autotrain as trainer
model = "F:\\spaces\\resources\\models\\ai-forever\\mGPT"
jsonl = "F:\\spaces\\resources\\datasets\\myblibible_data\\bibles\\kjv.bbl.jsonl"
result_file = "F:\\spaces\\resources\\datasets\\myblibible_data\\bibles\\kjv.bbl.trained"
config = {
  "chat_template": "none",
  "mixed_precision": "fp16",
  "optimizer": "adamw_torch",
  "peft": "true",
  "scheduler": "linear",
  "batch_size": "2",
  "block_size": "1024",
  "epochs": "3",
  "gradient_accumulation": "4",
  "lr": "0.00003",
  "model_max_length": "2048",
  "target_modules": "all-linear"
}
try:
    results = trainer.train(jsonl, result_file, model, config)
    print(f"results: {results}")
except Exception as e:
    print(f"results: {results}: {e}")