"""
Representation Tuning for privacy steering.

Fine-tunes attention weights (LoRA on V/O projections) with cosine loss:
    loss_l = (1 - cos_sim(h_l, v_l)) / 2
so activations permanently align with the privacy direction.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset


@dataclass
class RepTuningConfig:
    """Configuration for Representation Tuning."""

    # Model
    model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"

    # Privacy directions (from probe reader)
    reader_dir: str = ""  # path to a saved ProbeReader
    reader_type: str = "probe"  # "probe" or "pca"

    # Target layers for cosine similarity loss
    target_layers: list[int] = field(default_factory=lambda: [10, 12, 14, 16, 18])
    top_k_layers: int = 5

    # Representation tuning hyperparameters
    direction_mode: str = "in"  # "in" = tune direction in, "out" = tune direction out
    cos_loss_weight: float = 1.0     # weight for cosine similarity loss
    token_loss_weight: float = 0.1   # weight for token-level KL loss (capability preservation)
    token_loss_threshold: float = 0.7  # skip token loss if already below this (as in original blog)

    # CI-Decomposed Representation Tuning (Modification 4)
    # If ci_directions_dir is set, uses multi-objective CI-decomposed loss
    # instead of the single monolithic direction.
    ci_directions_dir: str = ""  # path to CI direction files from ci_decomposition.py
    ci_weights: dict = field(default_factory=dict)  # per-parameter loss weights

    # LoRA hyperparameters — target V and O projections (per blog finding)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["v_proj", "o_proj"]
    )

    # Training
    max_steps: int = 200
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    bf16: bool = True
    model_max_length: int = 512
    warmup_steps: int = 10
    output_dir: str = "outputs/rep_tuning"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


from ..utils.model_utils import format_chat_training as _format_chat_training


class RepTuningDataset(Dataset):
    """General instruction-following data for representation tuning (no contrastive pairs)."""

    def __init__(
        self,
        tokenizer,
        config: RepTuningConfig,
        instructions: Optional[list[str]] = None,
        responses: Optional[list[str]] = None,
        num_examples: int = 5000,
    ):
        self.tokenizer = tokenizer
        self.config = config

        if instructions is not None and responses is not None:
            self.instructions = instructions[:num_examples]
            self.responses = responses[:num_examples]
        else:
            self.instructions, self.responses = self._load_alpaca(num_examples)

        self.texts = []
        for instr, resp in zip(self.instructions, self.responses):
            resp_tokens = tokenizer.encode(resp, add_special_tokens=False)
            resp_truncated = tokenizer.decode(
                resp_tokens[:128], skip_special_tokens=True
            )
            self.texts.append(
                _format_chat_training(
                    tokenizer,
                    system_msg="You are a helpful assistant.",
                    user_msg=instr,
                    response=resp_truncated,
                )
            )

        print(f"  RepTuning dataset: {len(self)} samples")

    @staticmethod
    def _load_alpaca(num_examples: int) -> tuple[list[str], list[str]]:
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        ds = ds.filter(lambda x: x["input"] == "")
        instructions = ds["instruction"][:num_examples]
        outputs = ds["output"][:num_examples]
        return instructions, outputs

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokenized = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.config.model_max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": tokenized["input_ids"].squeeze(0).clone(),
        }


def _compute_cos_loss_single(
    hidden_states,
    attention_mask,
    target_layers: list[int],
    directions: dict[int, torch.Tensor],
    direction_mode: str,
) -> torch.Tensor:
    """Cosine similarity loss averaged over target layers for one set of directions."""
    cos_losses = []
    for layer_idx in target_layers:
        if layer_idx not in directions:
            continue

        direction = directions[layer_idx]
        h = hidden_states[layer_idx + 1]  # (batch, seq_len, hidden_dim)

        mask = attention_mask.unsqueeze(-1).float()
        h_masked = h * mask
        seq_lengths = mask.sum(dim=1).clamp(min=1)
        h_avg = h_masked.sum(dim=1) / seq_lengths

        direction_dev = direction.to(device=h_avg.device, dtype=h_avg.dtype)
        cos_sim = F.cosine_similarity(
            h_avg,
            direction_dev.unsqueeze(0).expand(h_avg.shape[0], -1),
            dim=-1,
        )  # (batch,)

        if direction_mode == "in":
            layer_loss = ((1.0 - cos_sim) / 2.0).mean()
        else:
            layer_loss = cos_sim.abs().mean()

        cos_losses.append(layer_loss)

    if cos_losses:
        return torch.stack(cos_losses).mean()
    return torch.tensor(0.0, device=attention_mask.device)


def rep_tuning_compute_loss(
    trainer_self,
    model,
    inputs,
    target_layers: list[int],
    privacy_directions: dict[int, torch.Tensor],
    direction_mode: str = "in",
    cos_loss_weight: float = 1.0,
    token_loss_weight: float = 0.1,
    token_loss_threshold: float = 0.7,
    return_outputs: bool = False,
    ci_directions: Optional[dict[str, dict[int, torch.Tensor]]] = None,
    ci_weights: Optional[dict[str, float]] = None,
    **kwargs,
):
    """Cosine-alignment loss, optionally multi-objective (CI-decomposed).

    Standard mode uses a single privacy direction; CI-decomposed mode
    sums weighted cosine losses over per-CI-parameter directions.
    An optional token-level CE loss preserves general capabilities.
    """
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    labels = inputs.get("labels", input_ids.clone())

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    hidden_states = outputs.hidden_states

    if ci_directions is not None:
        ci_w = ci_weights or {
            param: 1.0 / len(ci_directions) for param in ci_directions
        }
        cos_loss = torch.tensor(0.0, device=input_ids.device)
        for param_name, param_dirs in ci_directions.items():
            param_loss = _compute_cos_loss_single(
                hidden_states, attention_mask, target_layers,
                param_dirs, direction_mode,
            )
            w = ci_w.get(param_name, 1.0 / len(ci_directions))
            cos_loss = cos_loss + w * param_loss
    else:
        cos_loss = _compute_cos_loss_single(
            hidden_states, attention_mask, target_layers,
            privacy_directions, direction_mode,
        )

    token_loss = torch.tensor(0.0, device=input_ids.device)
    if token_loss_weight > 0:
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    if (token_loss < token_loss_threshold).item():
        total_loss = cos_loss_weight * cos_loss
    else:
        total_loss = cos_loss_weight * cos_loss + token_loss_weight * token_loss

    return (total_loss, outputs) if return_outputs else total_loss


class RepTuningTrainer:
    """Trains LoRA adapter with cosine-alignment loss to internalize privacy directions."""

    def __init__(self, config: RepTuningConfig):
        if LoraConfig is None:
            raise ImportError("peft package not installed. Run: pip install peft")

        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        self.privacy_directions = {}
        self.ci_directions: Optional[dict[str, dict[int, torch.Tensor]]] = None

    def _load_privacy_directions(self):
        """Load privacy directions from a saved reader."""
        reader_dir = Path(self.config.reader_dir)

        if self.config.reader_type == "probe":
            from ..reading.probe_reader import ProbeReader
            reader = ProbeReader()
            reader.load(str(reader_dir))
            best_layers = reader.get_best_layers(top_k=self.config.top_k_layers)

            for layer in best_layers:
                direction = reader.get_privacy_direction(layer)
                # Negate: probe direction points toward "appropriate" (sharing OK)
                self.privacy_directions[layer] = torch.from_numpy(-direction).float()

            self.config.target_layers = sorted(best_layers)
            print(f"  Loaded probe privacy directions for layers: {best_layers}")

        elif self.config.reader_type == "pca":
            from ..reading.pca_reader import PCAReader
            reader = PCAReader()
            reader.load(str(reader_dir))
            best_layers = reader.get_best_layers(top_k=self.config.top_k_layers)

            for layer in best_layers:
                direction = reader.get_privacy_vector(layer)
                self.privacy_directions[layer] = torch.from_numpy(-direction).float()

            self.config.target_layers = sorted(best_layers)
            print(f"  Loaded PCA privacy directions for layers: {best_layers}")

        else:
            raise ValueError(f"Unknown reader_type: {self.config.reader_type}")

        for layer, d in self.privacy_directions.items():
            print(f"    Layer {layer}: direction norm = {d.norm():.4f}")

    def _load_ci_directions(self):
        """Load CI-decomposed directions from ci_decomposition.py output."""
        ci_dir = Path(self.config.ci_directions_dir)
        if not ci_dir.exists():
            print(f"  WARNING: CI directions dir not found: {ci_dir}")
            return

        ci_params = ("info_type", "recipient", "transmission_principle")
        self.ci_directions = {}

        for param_name in ci_params:
            fname = ci_dir / f"ci_{param_name}_directions.pt"
            if not fname.exists():
                print(f"  Skipping CI param {param_name} (not found: {fname})")
                continue

            raw_dirs = torch.load(fname, map_location="cpu", weights_only=True)
            layer_dirs = {}
            for k, v in raw_dirs.items():
                layer_idx = int(k) if isinstance(k, str) else k
                # Only keep directions for target layers
                if layer_idx in self.config.target_layers:
                    # ci_decomposition.py saves with deterministic sign — no negation needed
                    layer_dirs[layer_idx] = v.float()
            self.ci_directions[param_name] = layer_dirs
            print(f"  Loaded CI direction '{param_name}': "
                  f"{len(layer_dirs)} layers")

        if not self.ci_directions:
            print("  WARNING: No CI directions loaded, "
                  "falling back to standard mode")
            self.ci_directions = None
        else:
            print(f"  CI-Decomposed mode: {list(self.ci_directions.keys())}")
            if self.config.ci_weights:
                print(f"  CI weights: {self.config.ci_weights}")

    def setup(self):
        """Load model, tokenizer, privacy directions, and prepare LoRA."""
        print(f"Loading model: {self.config.model_name_or_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            padding_side="left",
            model_max_length=self.config.model_max_length,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        compute_dtype = torch.bfloat16 if self.config.bf16 else torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=compute_dtype,
            device_map="auto",
        )

        print("\nLoading privacy directions...")
        self._load_privacy_directions()

        if self.config.ci_directions_dir:
            print("\nLoading CI-decomposed directions...")
            self._load_ci_directions()

        max_target = max(self.config.target_layers)
        lora_layers = list(range(max_target + 1))

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            layers_to_transform=lora_layers,
            task_type="CAUSAL_LM",
        )

        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()

        print(f"\n  Target layers: {self.config.target_layers}")
        print(f"  LoRA on layers 0..{max_target}")
        print(f"  LoRA modules: {self.config.lora_target_modules}")
        print(f"  Direction mode: {self.config.direction_mode}")
        print(f"  Cos loss weight: {self.config.cos_loss_weight}")
        print(f"  Token loss weight: {self.config.token_loss_weight}")

    def train(
        self,
        instructions: Optional[list[str]] = None,
        responses: Optional[list[str]] = None,
        num_examples: int = 5000,
    ):
        """Run Representation Tuning."""
        if self.peft_model is None:
            self.setup()

        dataset = RepTuningDataset(
            tokenizer=self.tokenizer,
            config=self.config,
            instructions=instructions,
            responses=responses,
            num_examples=num_examples,
        )

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            max_steps=self.config.max_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=self.config.warmup_steps,
            bf16=self.config.bf16,
            logging_steps=10,
            save_total_limit=0,
            report_to="none",
            remove_unused_columns=False,
            gradient_checkpointing=True,
            weight_decay=0.01,
        )

        target_layers = self.config.target_layers
        privacy_directions = self.privacy_directions
        direction_mode = self.config.direction_mode
        cos_loss_weight = self.config.cos_loss_weight
        token_loss_weight = self.config.token_loss_weight
        token_loss_threshold = self.config.token_loss_threshold
        ci_directions = self.ci_directions
        ci_weights = self.config.ci_weights or None

        class RepTuningCustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                return rep_tuning_compute_loss(
                    self,
                    model,
                    inputs,
                    target_layers=target_layers,
                    privacy_directions=privacy_directions,
                    direction_mode=direction_mode,
                    cos_loss_weight=cos_loss_weight,
                    token_loss_weight=token_loss_weight,
                    token_loss_threshold=token_loss_threshold,
                    return_outputs=return_outputs,
                    ci_directions=ci_directions,
                    ci_weights=ci_weights,
                )

        trainer = RepTuningCustomTrainer(
            model=self.peft_model,
            processing_class=self.tokenizer,
            args=training_args,
            train_dataset=dataset,
        )

        self.peft_model.config.use_cache = False
        print("\nStarting Representation Tuning...")
        trainer.train()
        print("Representation Tuning complete!")

    def save_merged_model(self, output_dir: str) -> Path:
        """Merge LoRA adapter into base model and save."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Merging LoRA adapter and saving to {output_path}...")
        merged_model = self.peft_model.merge_and_unload()
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(output_path / "rep_tuning_config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        metadata = {
            "target_layers": self.config.target_layers,
            "direction_mode": self.config.direction_mode,
            "reader_dir": self.config.reader_dir,
            "reader_type": self.config.reader_type,
            "direction_norms": {
                str(k): float(v.norm()) for k, v in self.privacy_directions.items()
            },
        }
        with open(output_path / "rep_tuning_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved merged model to {output_path}")
        return output_path

    def save_adapter(self, output_dir: str) -> Path:
        """Save just the LoRA adapter (without merging)."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.peft_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(output_path / "rep_tuning_config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        print(f"  Saved LoRA adapter to {output_path}")
        return output_path


def evaluate_robustness(
    model_helper,
    privacy_directions: dict[int, torch.Tensor],
    eval_prompts: list[str],
    alpha_values: list[float] = [-1.0, -2.0, -3.0],
    max_new_tokens: int = 256,
    batch_size: int = 8,
) -> dict:
    """Test if a tuned model resists negative (anti-privacy) steering."""
    from .steering import PrivacySteering

    results = {}

    baseline_outputs = model_helper.generate(
        texts=eval_prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    results["baseline"] = {
        "alpha": 0.0,
        "outputs": baseline_outputs,
    }

    for alpha in alpha_values:
        steerer = PrivacySteering(
            model_helper=model_helper,
            privacy_directions=privacy_directions,
            alpha=alpha,
            normalize=True,
        )
        steered_outputs = steerer.generate(
            prompts=eval_prompts,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )
        results[f"alpha_{alpha}"] = {
            "alpha": alpha,
            "outputs": steered_outputs,
        }

    return results
