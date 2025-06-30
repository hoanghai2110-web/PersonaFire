
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import os
from tqdm import tqdm
from typing import Dict, Optional, List

from model import UltraChronoFireTransformer
from components import UltraChronoFireOptimizer, ToneDetector
from custom_tokenizer import UltraChronoFireTokenizer, train_custom_tokenizer

class ConversationDataset(Dataset):
    """Dataset cho dữ liệu hội thoại với emotion và style"""
    
    def __init__(self, jsonl_path: str, tokenizer_path: str = "./model/ultra_tokenizer.pkl", 
                 max_length: int = 512):
        # Load custom tokenizer
        if os.path.exists(tokenizer_path):
            self.tokenizer = UltraChronoFireTokenizer.load(tokenizer_path)
        else:
            print("⚠️ Custom tokenizer not found, training new one...")
            self.tokenizer = train_custom_tokenizer(jsonl_path)
        
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.max_length = max_length
        
        # Load data
        self.conversations = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                self.conversations.append(data)
        
        print(f"✅ Loaded {len(self.conversations)} conversations")
        
        # Debug: Check token ranges in first few samples
        print("🔍 Checking token ranges...")
        for i, conv in enumerate(self.conversations[:3]):
            user_tokens = self.tokenizer.encode(conv.get('input', ''))
            assistant_tokens = self.tokenizer.encode(conv.get('output', ''))
            all_tokens = user_tokens + assistant_tokens
            
            if all_tokens:
                max_token = max(all_tokens)
                min_token = min(all_tokens)
                print(f"  Sample {i}: tokens range [{min_token}, {max_token}], vocab_size: {self.tokenizer.vocab_size_actual}")
                
                if max_token >= self.tokenizer.vocab_size_actual:
                    print(f"  ⚠️ Warning: Token {max_token} >= vocab_size {self.tokenizer.vocab_size_actual}")
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx: int) -> Dict:
        conv = self.conversations[idx]
        
        # Extract conversation parts
        user_input = conv.get('input', '')
        assistant_response = conv.get('output', '')
        
        # Get tone and style
        tone_id = ToneDetector.get_tone_id(conv.get('tone', 'neutral'))
        style_id = conv.get('style_id', 1)
        
        # Prepare input text
        user_tokens = self.tokenizer.encode(user_input)
        assistant_tokens = self.tokenizer.encode(assistant_response)
        
        # Combine tokens
        full_tokens = user_tokens + assistant_tokens
        
        # Validate token IDs - clamp to vocab size
        max_token_id = self.tokenizer.vocab_size_actual - 1
        full_tokens = [min(token, max_token_id) for token in full_tokens]
        
        # Truncate and pad
        if len(full_tokens) > self.max_length:
            full_tokens = full_tokens[:self.max_length]
        
        # Create attention mask
        attention_mask = [1] * len(full_tokens)
        
        # Pad to max_length
        while len(full_tokens) < self.max_length:
            full_tokens.append(self.pad_token_id)
            attention_mask.append(0)
        
        input_ids = torch.tensor(full_tokens, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        # Create labels (shift by 1 for next token prediction)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding in loss
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'tone_id': torch.tensor(tone_id, dtype=torch.long),
            'style_id': torch.tensor(style_id, dtype=torch.long),
            'user_text': user_input,
            'assistant_text': assistant_response
        }
        
        # Add emotion labels if available
        if 'emotion' in conv:
            emotion_labels = conv['emotion']
            if isinstance(emotion_labels, list) and len(emotion_labels) >= 4:
                # Ensure 8D emotion vector
                if len(emotion_labels) == 4:
                    # Expand [sad, happy, fear, anger] to 8D
                    expanded_emotion = [
                        emotion_labels[0],  # sad
                        emotion_labels[1],  # happy
                        emotion_labels[2],  # fear
                        emotion_labels[3],  # anger
                        0.1,  # surprise
                        0.1,  # disgust
                        0.2,  # neutral
                        emotion_labels[1] * 0.8  # joy
                    ]
                    result['emotion_labels'] = torch.tensor(expanded_emotion, dtype=torch.float32)
                else:
                    result['emotion_labels'] = torch.tensor(emotion_labels[:8], dtype=torch.float32)
        
        return result

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    tone_ids = torch.stack([item['tone_id'] for item in batch])
    style_ids = torch.stack([item['style_id'] for item in batch])
    
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'tone_ids': tone_ids,
        'style_ids': style_ids,
        'user_texts': [item['user_text'] for item in batch],
        'assistant_texts': [item['assistant_text'] for item in batch]
    }
    
    # Add emotion labels if available
    if 'emotion_labels' in batch[0]:
        emotion_labels = torch.stack([item['emotion_labels'] for item in batch])
        result['emotion_labels'] = emotion_labels
    
    return result

class UltraChronoFireTrainer:
    """Trainer for Ultra ChronoFire Transformer - T4 Optimized"""
    
    def __init__(self, model: UltraChronoFireTransformer, 
                 train_dataloader: DataLoader,
                 learning_rate: float = 2e-4,
                 weight_decay: float = 0.1,
                 warmup_steps: int = 1000,
                 max_steps: int = 20000,
                 save_dir: str = "./checkpoints",
                 emotion_loss_weight: float = 0.1,
                 gradient_accumulation_steps: int = 1,
                 use_mixed_precision: bool = False,
                 max_grad_norm: float = 1.0):
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_mixed_precision = use_mixed_precision
        self.max_grad_norm = max_grad_norm
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
        
        # Optimizer and scheduler
        self.optimizer = UltraChronoFireOptimizer.create_optimizer(
            model, lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = UltraChronoFireOptimizer.get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps=warmup_steps, total_steps=max_steps
        )
        
        self.max_steps = max_steps
        self.save_dir = save_dir
        self.emotion_loss_weight = emotion_loss_weight
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        self.step = 0
        self.best_loss = float('inf')
        self.accumulated_loss = 0.0
    
    def compute_loss(self, batch: Dict, outputs: Dict) -> Dict:
        """Compute total loss including emotion consistency"""
        logits = outputs['logits']
        labels = batch['labels']
        
        # Standard language modeling loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        total_loss = lm_loss
        loss_dict = {'lm_loss': lm_loss.item()}
        
        # Emotion consistency loss
        if 'emotion_states' in outputs and 'emotion_labels' in batch:
            emotion_loss = self.model.compute_emotion_consistency_loss(
                outputs['emotion_states'], 
                batch['emotion_labels']
            )
            total_loss += self.emotion_loss_weight * emotion_loss
            loss_dict['emotion_loss'] = emotion_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict
    
    def train_step(self, batch: Dict, accumulation_step: int) -> Dict:
        """Single training step with gradient accumulation and mixed precision"""
        self.model.train()
        
        # Move batch to device
        device = next(self.model.parameters()).device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device, non_blocking=True)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
            input_ids = batch['input_ids']
            input_text = batch['user_texts'][0] if batch['user_texts'] else ""
            
            logits, states = self.model(
                input_ids,
                tone_override=batch.get('tone_ids'),
                style_id=batch.get('style_ids'),
                input_text=input_text
            )
            
            # Collect emotion states for loss
            emotion_states = []
            for state in states[:-1]:  # Exclude the last state
                if 'emotion' in state:
                    emotion_states.append(state['emotion'])
            
            outputs = {
                'logits': logits,
                'emotion_states': emotion_states,
                'states': states
            }
            
            # Compute loss and scale for gradient accumulation
            loss, loss_dict = self.compute_loss(batch, outputs)
            loss = loss / self.gradient_accumulation_steps
        
        # Backward pass with mixed precision
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Accumulate loss for logging
        self.accumulated_loss += loss_dict['total_loss']
        
        # Update weights only on accumulation boundary
        if (accumulation_step + 1) % self.gradient_accumulation_steps == 0:
            if self.use_mixed_precision:
                # Gradient clipping with scaler
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard gradient clipping and step
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            # Return accumulated loss
            loss_dict['total_loss'] = self.accumulated_loss / self.gradient_accumulation_steps
            loss_dict['lr'] = self.scheduler.get_last_lr()[0]
            self.accumulated_loss = 0.0
        else:
            # Don't update LR during accumulation
            loss_dict['lr'] = self.scheduler.get_last_lr()[0]
        
        return loss_dict
    
    def save_checkpoint(self, step: int, loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'model_config': {
                'vocab_size': self.model.token_embedding.num_embeddings,
                'd_model': self.model.d_model,
                'num_layers': self.model.num_layers,
                'num_heads': self.model.layers[0].num_heads if self.model.layers else 8,
                'k_window': self.model.k_window
            }
        }
        
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_step_{step}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"🏆 Best model saved: {best_path}")
    
    def train(self, save_every: int = 500, log_every: int = 50):
        """Main training loop with gradient accumulation"""
        print("🚀 Starting Ultra ChronoFire training (T4 Optimized)...")
        print(f"📊 Model stats: {self.model.get_performance_stats()}")
        print(f"🔧 Mixed precision: {self.use_mixed_precision}")
        print(f"🔧 Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"🔧 Effective batch size: {self.train_dataloader.batch_size * self.gradient_accumulation_steps}")
        
        self.model.train()
        device = next(self.model.parameters()).device
        print(f"🔥 Training on device: {device}")
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        pbar = tqdm(total=self.max_steps, desc="Training")
        accumulation_step = 0
        
        while self.step < self.max_steps:
            for batch in self.train_dataloader:
                if self.step >= self.max_steps:
                    break
                
                # Training step with accumulation
                loss_dict = self.train_step(batch, accumulation_step)
                accumulation_step += 1
                
                # Only increment step and log when we actually update weights
                if accumulation_step % self.gradient_accumulation_steps == 0:
                    self.step += 1
                    
                    # Logging
                    if self.step % log_every == 0:
                        log_str = f"Step {self.step}"
                        for key, value in loss_dict.items():
                            log_str += f" | {key}: {value:.4f}"
                        
                        # Add GPU memory usage
                        if torch.cuda.is_available():
                            memory_used = torch.cuda.memory_allocated() / 1024**3
                            log_str += f" | GPU: {memory_used:.1f}GB"
                        
                        pbar.set_description(log_str)
                    
                    # Save checkpoint
                    if self.step % save_every == 0 and self.step > 0:
                        self.save_checkpoint(self.step, loss_dict['total_loss'])
                        # Clear cache after saving
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    pbar.update(1)
        
        pbar.close()
        
        # Final checkpoint
        if self.step > 0:
            self.save_checkpoint(self.step, loss_dict.get('total_loss', 0.0))
        
        print("🎉 Training completed!")

def main():
    """Main training function"""
    
    print("🔥 Ultra ChronoFire Training Script")
    print("=" * 50)
    
    # Configuration - optimized for T4 GPU
    config = {
        'vocab_size': 50000,  # Will be updated to actual size
        'd_model': 384,        # Further reduced for T4
        'num_layers': 4,       # Smaller for faster training
        'k_window': 8,         # Reduced memory window
        'num_heads': 6,        # Reduced heads
        'dropout': 0.1,
        'learning_rate': 2e-4, # Higher LR with gradient accumulation
        'weight_decay': 0.01,
        'batch_size': 8,       # Increased batch size for T4
        'gradient_accumulation_steps': 4,  # Effective batch size = 32
        'max_steps': 8000,     # Reduced steps
        'warmup_steps': 400,   # Reduced warmup
        'emotion_loss_weight': 0.1,
        'data_path': 'user_training_data.jsonl',
        'use_mixed_precision': True,  # Enable AMP
        'max_grad_norm': 0.5,  # Gradient clipping
        'save_every': 500,     # More frequent saves
        'log_every': 50        # More frequent logging
    }
    
    # Check if data exists
    if not os.path.exists(config['data_path']):
        print(f"❌ Training data not found: {config['data_path']}")
        return
    
    print(f"📚 Using training data: {config['data_path']}")
    
    # Load data
    print("🔄 Loading data...")
    dataset = ConversationDataset(config['data_path'], max_length=512)
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,  # Utilize CPU cores for data loading
        pin_memory=True,
        persistent_workers=True  # Keep workers alive
    )
    
    # Create model
    print("🏗️ Creating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get actual vocab size from tokenizer
    actual_vocab_size = dataset.tokenizer.vocab_size_actual
    print(f"📖 Actual vocabulary size: {actual_vocab_size}")
    
    # Update config to match actual vocab size
    config['vocab_size'] = actual_vocab_size
    
    model = UltraChronoFireTransformer(
        vocab_size=actual_vocab_size,
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        k_window=config['k_window'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        use_neuromodulation=True,
        use_kernelized_attention=True,
        personality_mode='fixed'
    ).to(device)
    
    # Set personality
    model.set_personality('creative')
    
    print(f"✅ Model created: {model.get_performance_stats()}")
    
    # Create trainer
    trainer = UltraChronoFireTrainer(
        model=model,
        train_dataloader=train_dataloader,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        warmup_steps=config['warmup_steps'],
        max_steps=config['max_steps'],
        emotion_loss_weight=config['emotion_loss_weight'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        use_mixed_precision=config['use_mixed_precision'],
        max_grad_norm=config['max_grad_norm']
    )
    
    # Start training with optimized settings
    trainer.train(
        save_every=config['save_every'],
        log_every=config['log_every']
    )
    
    print("🎯 Training finished! Use generate_chat.py to test the model.")

if __name__ == "__main__":
    main()
