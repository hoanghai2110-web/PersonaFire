import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Dict
from .components import *

class UltraChronoFireTransformer(nn.Module):
    """Ultra ChronoFire với tất cả advanced features"""

    def __init__(self, vocab_size: int, d_model: int = 768, num_layers: int = 12, 
                 k_window: int = 32, num_heads: int = 12, dropout: float = 0.1,
                 max_seq_len: int = 4096, use_neuromodulation: bool = True,
                 use_kernelized_attention: bool = True, personality_mode: str = 'adaptive',
                 num_style_ids: int = 8):
        super().__init__()
        self.d_model = d_model
        self.k_window = k_window
        self.num_layers = num_layers
        self.personality_mode = personality_mode
        self.num_style_ids = num_style_ids

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.style_embedding = nn.Embedding(num_style_ids, d_model // 4)
        self.style_projector = nn.Linear(d_model // 4, d_model)

        # Ultra ChronoFire layers
        self.layers = nn.ModuleList([
            UltraChronoFireBlock(
                d_model=d_model,
                k_window=k_window,
                num_heads=num_heads,
                dropout=dropout,
                use_neuromodulation=use_neuromodulation,
                use_kernelized_attention=use_kernelized_attention,
                sparsity_threshold=0.3 + 0.1 * i,  # Gradually increase sparsity in deeper layers
                layer_idx=i
            ) for i in range(num_layers)
        ])

        # Personality system với shift resistance
        if personality_mode == 'fixed':
            self.personality_profiles = {
                'aggressive': {
                    'emotion': torch.tensor([0.8, -0.2, 0.3, -0.5, 0.6, -0.1, 0.4, -0.3]),
                    'shift_resistance': 0.2  # Changes emotion quickly
                },
                'gentle': {
                    'emotion': torch.tensor([-0.3, 0.7, -0.2, -0.8, -0.4, 0.6, -0.1, 0.5]),
                    'shift_resistance': 0.8  # Very stable, hard to change
                },
                'analytical': {
                    'emotion': torch.tensor([0.1, -0.1, -0.6, 0.4, -0.2, -0.3, 0.8, 0.2]),
                    'shift_resistance': 0.6  # Moderate stability
                },
                'creative': {
                    'emotion': torch.tensor([0.2, 0.6, 0.1, -0.2, 0.7, 0.3, -0.1, 0.4]),
                    'shift_resistance': 0.3  # Changes fairly easily
                },
                'teasing': {
                    'emotion': torch.tensor([0.1, 0.8, 0.4, -0.1, 0.6, -0.2, 0.3, 0.7]),
                    'shift_resistance': 0.1  # Changes very quickly, playful
                },
                'cute': {
                    'emotion': torch.tensor([-0.1, 0.9, -0.3, -0.7, -0.2, 0.8, 0.2, 0.6]),
                    'shift_resistance': 0.4  # Moderately stable but responsive
                }
            }

        # Output components
        self.ln_final = nn.LayerNorm(d_model)
        self.response_polisher = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout * 0.5)  # Lighter dropout for response polishing
        )
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Tone/Contextual embedding system
        self.tone_vocab_size = 16  # vui, buồn, tò mò, nghiêm túc, etc.
        self.tone_embedding = nn.Embedding(self.tone_vocab_size, d_model // 8)
        self.tone_mixer = nn.Linear(d_model // 8, d_model)

        # Performance monitoring
        self.register_buffer('step_count', torch.tensor(0))

        self.dropout = nn.Dropout(dropout)

    def set_personality(self, personality: str):
        """Set fixed personality mode với shift resistance"""
        if self.personality_mode == 'fixed' and personality in self.personality_profiles:
            profile = self.personality_profiles[personality]
            emotion_vector = profile['emotion']
            shift_resistance = profile['shift_resistance']

            for layer in self.layers:
                if hasattr(layer, 'neuromodulation'):
                    layer.neuromodulation.personality.data = emotion_vector.to(layer.neuromodulation.personality.device)
                    layer.neuromodulation.shift_resistance.data = torch.tensor(shift_resistance).to(layer.neuromodulation.shift_resistance.device)

    def should_stay_silent(self, emotion_state: torch.Tensor, input_intensity: float = 1.0) -> torch.Tensor:
        """Determine if model should stay silent based on emotion"""
        # Use final layer's emotion for silence decision
        if hasattr(self.layers[-1], 'neuromodulation'):
            threshold = self.layers[-1].neuromodulation.silent_threshold

            # Calculate emotion intensity (how "overwhelmed" the model feels)
            emotion_intensity = torch.norm(emotion_state, dim=-1)

            # Strong negative emotions (sadness, fear) might trigger silence
            sad_fear_emotions = emotion_state[:, [0, 2]]  # Assuming indices 0,2 are sad/fear
            negative_intensity = torch.norm(sad_fear_emotions, dim=-1)

            # Silent if negative emotions are strong and overall intensity is low
            should_silent = (negative_intensity > threshold) & (emotion_intensity < 0.5)

            return should_silent.float()

        return torch.zeros(emotion_state.shape[0], device=emotion_state.device)

    def compute_emotion_consistency_loss(self, emotion_states: List[torch.Tensor], 
                                               target_emotions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute auxiliary loss for emotion consistency"""
        if not emotion_states or target_emotions is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        # Use final emotion state for loss
        final_emotion = emotion_states[-1]  # Last layer's emotion

        # MSE loss between predicted and target emotions
        emotion_loss = F.mse_loss(final_emotion, target_emotions)

        # Add emotion smoothness regularization (prevent wild swings)
        if len(emotion_states) > 1:
            emotion_diff = final_emotion - emotion_states[-2]
            smoothness_loss = 0.1 * torch.mean(emotion_diff ** 2)
            emotion_loss += smoothness_loss

        return emotion_loss

    def get_performance_stats(self) -> Dict:
        """Get model performance statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Calculate emotion decay stats across layers
        emotion_decays = []
        for layer in self.layers:
            if hasattr(layer, 'neuromodulation'):
                emotion_decays.append(layer.neuromodulation.emotion_decay.item())

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,  # Assuming float32
            'step_count': self.step_count.item(),
            'emotion_decay_range': f"{min(emotion_decays):.3f}-{max(emotion_decays):.3f}" if emotion_decays else "N/A",
            'tone_vocab_size': self.tone_vocab_size
        }

    def detect_tone_from_context(self, x: torch.Tensor) -> torch.Tensor:
        """Simple tone detection based on input patterns"""
        # This is a simplified version - in practice, you'd train this or use external tone classifier
        batch_size = x.shape[0]

        # Basic heuristic: use token embeddings to infer tone
        # In real implementation, you could use a separate tone classifier
        token_emb = self.token_embedding(x)
        tone_signal = token_emb.mean(dim=1)  # Average across sequence

        # Map to tone categories (0-15 for 16 different tones)
        tone_logits = torch.matmul(tone_signal, self.tone_embedding.weight.T)
        tone_ids = torch.argmax(tone_logits, dim=-1)

        return tone_ids

    def generate_response(self, input_ids: torch.Tensor, max_length: int = 50, 
                         temperature: float = 0.7, top_p: float = 0.9, 
                         personality: str = 'creative', input_text: str = "") -> str:
        """Generate response với emotion và personality"""
        self.eval()

        # Set personality
        if personality:
            self.set_personality(personality)

        device = input_ids.device
        batch_size = input_ids.shape[0]

        # Initialize states
        past_states = [{} for _ in range(self.num_layers)]
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits, past_states = self.forward(
                    generated, 
                    past_states=past_states,
                    input_text=input_text if len(generated[0]) == len(input_ids[0]) else None
                )

                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature

                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)

                # Check for EOS token (assuming 2 is EOS)
                if next_token.item() == 2:
                    break

        return generated

    def forward(self, x: torch.Tensor, past_states: Optional[List[Dict]] = None, 
                tone_override: Optional[torch.Tensor] = None,
                style_id: Optional[torch.Tensor] = None,
                input_text: Optional[str] = None) -> Tuple[torch.Tensor, List[Dict]]:
        batch_size, seq_len = x.shape
        device = x.device

        # Embeddings
        token_emb = self.token_embedding(x)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos_ids)

        # Style embedding (for multi-style training)
        if style_id is not None:
            style_emb = self.style_embedding(style_id)
            style_signal = self.style_projector(style_emb).unsqueeze(1)
        else:
            style_signal = torch.zeros_like(token_emb[:, :1])

        # Auto tone detection from text if provided
        if tone_override is not None:
            tone_ids = tone_override
        elif input_text is not None:
            # Auto detect tone from actual text
            from .components import ToneDetector
            tone_name = ToneDetector.detect_tone_from_text(input_text)
            tone_id = ToneDetector.get_tone_id(tone_name)
            tone_ids = torch.full((batch_size,), tone_id, device=device, dtype=torch.long)
        else:
            tone_ids = self.detect_tone_from_context(x)

        tone_emb = self.tone_embedding(tone_ids)
        tone_signal = self.tone_mixer(tone_emb).unsqueeze(1)  # Broadcast across sequence

        # Combine all embeddings
        hidden_states = self.dropout(token_emb + pos_emb + 0.1 * tone_signal + 0.05 * style_signal)

        # Initialize states
        if past_states is None:
            past_states = [{} for _ in range(self.num_layers)]

        # Process each token
        outputs = []
        new_states = []

        for t in range(seq_len):
            x_t = hidden_states[:, t]
            layer_states = []

            for i, layer in enumerate(self.layers):
                x_t, state = layer(x_t, past_states[i] if i < len(past_states) else {})
                layer_states.append(state)

            outputs.append(x_t)
            if t == seq_len - 1:  # Only keep states from last timestep
                new_states = layer_states

        # Final processing
        hidden_states = torch.stack(outputs, dim=1)
        hidden_states = self.ln_final(hidden_states)

        # Apply response polisher for smoother language
        polished_states = self.response_polisher(hidden_states)

        # Boost emotion into output logits
        final_emotion_state = new_states[-1].get('emotion', torch.zeros(batch_size, 8, device=device))

        # Check if model should stay silent
        should_silent = self.should_stay_silent(final_emotion_state)

        if hasattr(self.layers[-1], 'neuromodulation'):
            emotion_embed = self.layers[-1].neuromodulation.emotion_mixer(final_emotion_state)
            # Apply emotion boost to each position in sequence
            emotion_boost = 0.15 * emotion_embed.unsqueeze(1)  # Broadcast to sequence length
            enhanced_states = polished_states + emotion_boost
        else:
            enhanced_states = polished_states

        logits = self.output_proj(enhanced_states)

        # Apply silent emotion masking - boost probability of silence tokens
        if should_silent.any():
            # Assuming tokens for silence: "...", ".", "🥺" have specific IDs
            # This would need to be configured based on actual tokenizer
            silence_token_ids = [1, 2, 3]  # Placeholder - replace with actual IDs
            for token_id in silence_token_ids:
                logits[:, :, token_id] = logits[:, :, token_id] + should_silent.unsqueeze(1) * 2.0

        # Store silence flag in states for monitoring
        new_states.append({'should_silent': should_silent.cpu().numpy().tolist()})

        # Update step count
        self.step_count += 1

        return logits, new_states

# Demo and testing
def demo_ultra_chronofire():
    """Demo Ultra ChronoFire với tất cả features"""
    print("🚀 Ultra ChronoFire Transformer Demo - Emotional Intelligence Peak")
    print("=" * 60)

    # Model configuration
    model = UltraChronoFireTransformer(
        vocab_size=50000,
        d_model=768,
        num_layers=8,  # Smaller for demo
        k_window=16,
        num_heads=12,
        dropout=0.1,
        use_neuromodulation=True,
        use_kernelized_attention=True,
        personality_mode='fixed',
        num_style_ids=8
    )

    # Set personality
    model.set_personality('creative')
    print("🎭 Personality set to: Creative")

    # Sample input
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, 50000, (batch_size, seq_len))

    print(f"\n📊 Model Stats:")
    stats = model.get_performance_stats()
    for key, value in stats.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value:.2f}")

    # Forward pass
    print(f"\n🔄 Processing sequence: {input_ids.shape}")
    with torch.no_grad():
        # Test with different tone overrides
        logits, states = model(input_ids)

        # Test with specific tone (e.g., happy = 0, sad = 1, curious = 2)
        happy_tone = torch.zeros(batch_size, dtype=torch.long)  # Happy tone
        logits_happy, _ = model(input_ids, tone_override=happy_tone)

        sad_tone = torch.ones(batch_size, dtype=torch.long)  # Sad tone
        logits_sad, _ = model(input_ids, tone_override=sad_tone)

    print(f"✅ Output logits shape: {logits.shape}")
    print(f"🧠 Number of layer states: {len(states)}")
    print(f"🎭 Tone impact: Happy vs Sad logits diff = {(logits_happy - logits_sad).abs().mean().item():.4f}")

    # Analyze states
    if states:
        sample_state = states[0]
        print(f"\n🔍 Layer 0 state analysis:")
        for key, value in sample_state.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, list):
                print(f"  {key}: list with {len(value)} items")
            else:
                print(f"  {key}: {value}")

    # Test emotion consistency loss
    emotion_states = [state.get('emotion') for state in states if state.get('emotion') is not None]
    if emotion_states:
        # Create dummy target emotions
        target_emotions = torch.softmax(torch.randn(batch_size, 8), dim=-1)
        emotion_loss = model.compute_emotion_consistency_loss(emotion_states, target_emotions)
        print(f"\n💭 Emotion consistency loss: {emotion_loss.item():.4f}")

    # Create optimizer
    optimizer = UltraChronoFireOptimizer.create_optimizer(model)
    scheduler = UltraChronoFireOptimizer.get_cosine_schedule_with_warmup(optimizer)

    print(f"\n⚡ Optimizer created with {len(optimizer.param_groups)} parameter groups")
    for i, group in enumerate(optimizer.param_groups):
        print(f"  Group {i}: {len(group['params'])} params, lr={group['lr']:.4f}")

    print(f"\n🎯 Ultra ChronoFire ready for training!")
    print(f"💡 Key features enabled:")
    print(f"  ✓ Dilated Temporal Sampling")
    print(f"  ✓ Neuromodulation System with Layer-specific Emotion Decay") 
    print(f"  ✓ Kernelized Attention (O(n) complexity)")
    print(f"  ✓ Per-head Memory States")
    print(f"  ✓ Adaptive Sparsity")
    print(f"  ✓ Personality Modes with Emotion Shift Resistance")
    print(f"  ✓ Emotion-Boosted Output Logits")
    print(f"  ✓ Response Polisher for Smoother Language")
    print(f"  ✓ Auto Tone Detection from Text")
    print(f"  ✓ Silent Emotion System (can stay quiet when overwhelmed)")
    print(f"  ✓ Multi-Style Training Support")
    print(f"  ✓ Contextual Tone Embedding System")
    print(f"  ✓ Emotion Consistency Loss")

    return model, optimizer, scheduler

if __name__ == "__main__":
    # Run demo
    model, optimizer, scheduler = demo_ultra_chronofire()

    # Test auto tone detection
    print(f"\n🎭 Testing auto tone detection...")
    test_texts = [
        "Tôi đang buồn quá...",
        "Haha bạn đùa hay thật đấy!",
        "Ơi bạn iu ơi!",
        "Tôi tức quá!",
        "Tại sao lại như vậy?",
        "Xin chào bạn."
    ]

    for text in test_texts:
        from .components import ToneDetector
        detected_tone = ToneDetector.detect_tone_from_text(text)
        tone_id = ToneDetector.get_tone_id(detected_tone)
        print(f"  '{text}' → {detected_tone} (ID: {tone_id})")

    # Test personalities with shift resistance
    print(f"\n🧪 Testing personalities with emotion shift resistance...")
    personalities = ['aggressive', 'gentle', 'analytical', 'creative', 'teasing', 'cute']

    sample_input = torch.randint(0, 50000, (1, 32))

    for personality in personalities:
        model.set_personality(personality)

        # Get resistance value
        resistance = model.personality_profiles[personality]['shift_resistance']

        with torch.no_grad():
            logits, states = model(sample_input, input_text="Tôi buồn quá...")

        # Check if silence was triggered
        should_silent = states[-1].get('should_silent', [False])[0] if states else False

        # Analyze output distribution
        probs = F.softmax(logits[0, -1], dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum().item()

        print(f"  {personality:>12}: resistance={resistance:.1f}, entropy={entropy:.3f}, silent={should_silent}")

    # Test style embeddings
    print(f"\n🎨 Testing style system...")
    style_names = ['formal', 'casual', 'cute', 'professional', 'friendly', 'mysterious', 'energetic', 'calm']

    for style_id, style_name in enumerate(style_names):
        style_tensor = torch.tensor([style_id], dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(sample_input, style_id=style_tensor, input_text="Hello!")

        probs = F.softmax(logits[0, -1], dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum().item()

        print(f"  {style_name:>12} (style {style_id}): entropy={entropy:.3f}")

    # Test silent emotion system
    print(f"\n🤫 Testing silent emotion system...")
    model.set_personality('gentle')  # Gentle personality for testing silence

    sad_texts = ["Tôi buồn lắm...", "😢", "..."]
    for text in sad_texts:
        with torch.no_grad():
            logits, states = model(sample_input, input_text=text)

        should_silent = states[-1].get('should_silent', [False])[0] if states else False
        emotion_state = states[0].get('emotion', torch.zeros(8)) if states else torch.zeros(8)
        emotion_intensity = torch.norm(emotion_state).item()

        print(f"  '{text}' → silent: {should_silent}, emotion_intensity: {emotion_intensity:.3f}")

    print(f"\n🌟 Ultra ChronoFire: Peak Emotional Intelligence Achieved!")
    print(f"🚀 Now with human-like emotional responses!")
    print(f"   • Emotion shift resistance prevents manipulation")
    print(f"   • Auto tone detection from natural text patterns")  
    print(f"   • Silent emotion system - can be overwhelmed and stay quiet")
    print(f"   • Multi-style training for different personalities")
    print(f"   • Dynamic emotion adaptation with personality constraints")
    print(f"   • Natural emotional responses like real humans! 🥺😊😡")