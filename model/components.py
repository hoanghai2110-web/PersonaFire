import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Dict

class DilatedTemporalSampler:
    """Shifted Windowing / Dilated Temporal Span - lấy mẫu cách đoạn để nhìn xa hơn"""

    @staticmethod
    def get_dilated_indices(current_idx: int, max_dilation: int = 8, num_samples: int = 8) -> List[int]:
        """
        Lấy indices theo pattern: t-1, t-2, t-4, t-8, ...
        Tương tự Dilated CNN nhưng cho temporal dimension
        """
        indices = []
        dilation = 1

        for _ in range(num_samples):
            idx = current_idx - dilation
            if idx >= 0:
                indices.append(idx)
            dilation = min(dilation * 2, max_dilation)

        return indices[::-1]  # Reverse để có thứ tự thời gian

class NeuromodulationUnit(nn.Module):
    """Neuromodulation - Memory chịu ảnh hưởng bởi cảm xúc/tâm trạng"""

    def __init__(self, d_model: int, layer_idx: int = 0, base_emotion_decay: float = 0.9):
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        # Deeper layers have longer emotion memory (higher decay)
        self.emotion_decay = nn.Parameter(torch.tensor(base_emotion_decay + 0.03 * layer_idx))
        self.emotion_decay.data.clamp_(0.7, 0.98)  # Keep in reasonable range

        # Emotion processing
        self.W_emotion = nn.Linear(d_model, d_model // 4)
        self.emotion_classifier = nn.Linear(d_model // 4, 8)  # 8 loại cảm xúc cơ bản
        self.emotion_mixer = nn.Linear(8, d_model)

        # Emotion shift resistance system
        self.shift_resistance = nn.Parameter(torch.tensor(0.3))  # Default resistance
        self.resistance_gate = nn.Linear(d_model, 1)  # Dynamic resistance based on input

        # Memory modulation
        self.memory_modulator = nn.Linear(d_model * 2, d_model)

        # Personality traits (có thể học được qua training)
        self.personality = nn.Parameter(torch.randn(8) * 0.1)  # anger, joy, fear, etc.

        # Silent emotion threshold
        self.silent_threshold = nn.Parameter(torch.tensor(0.2))  # When to stay silent

    def forward(self, x_t: torch.Tensor, prev_emotion: torch.Tensor, memory_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x_t.shape[0]

        # Extract emotion từ input hiện tại
        emotion_features = torch.tanh(self.W_emotion(x_t))
        emotion_logits = self.emotion_classifier(emotion_features)

        # Combine với personality và previous emotion
        proposed_emotion = F.softmax(emotion_logits + self.personality.unsqueeze(0), dim=-1)

        # Dynamic shift resistance based on input intensity
        dynamic_resistance = torch.sigmoid(self.resistance_gate(x_t)) * self.shift_resistance

        # Apply emotion shift resistance - harder to change personality
        emotion_state = (1 - dynamic_resistance) * proposed_emotion + dynamic_resistance * prev_emotion

        # Temporal emotion decay (still keep some decay for natural evolution)
        emotion_state = self.emotion_decay * prev_emotion + (1 - self.emotion_decay) * emotion_state

        # Modulate memory based on emotion
        emotion_signal = self.emotion_mixer(emotion_state)
        modulation_input = torch.cat([memory_state, emotion_signal], dim=-1)
        memory_gate = torch.sigmoid(self.memory_modulator(modulation_input))

        # Apply emotional modulation (1 + emotion cho phép tăng/giảm linh hoạt)
        modulated_memory = memory_state * (1 + 0.2 * torch.tanh(emotion_signal))

        return modulated_memory * memory_gate, emotion_state

class KernelizedAttention(nn.Module):
    """Kernel-based attention (Fastformer style) - O(n) complexity"""

    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Kernel projections
        self.W_q_kernel = nn.Linear(d_model, d_model)
        self.W_k_kernel = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Normalization factors
        self.kernel_alpha = nn.Parameter(torch.ones(num_heads))

    def kernel_fn(self, x: torch.Tensor) -> torch.Tensor:
        """ReLU kernel - có thể thay bằng ELU, GELU, etc."""
        return F.relu(x) + 1e-6  # Thêm epsilon cho stability

    def forward(self, q: torch.Tensor, k_history: List[torch.Tensor], v_history: List[torch.Tensor]) -> torch.Tensor:
        batch_size = q.shape[0]

        if not k_history:
            return self.W_o(q)

        # Kernel projections
        q_kernel = self.kernel_fn(self.W_q_kernel(q))  # [batch, d_model]

        # Process history
        k_kernels = [self.kernel_fn(self.W_k_kernel(k)) for k in k_history]
        v_projected = [self.W_v(v) for v in v_history]

        # Reshape for multi-head
        q_kernel = q_kernel.view(batch_size, self.num_heads, self.head_dim)

        # Efficient kernel attention: sum(q) * sum(k * v) / normalizer
        outputs = []
        for h in range(self.num_heads):
            q_h = q_kernel[:, h]  # [batch, head_dim]

            # Compute sum(k * v) for each head
            kv_sum = torch.zeros_like(q_h)
            k_sum = torch.zeros(batch_size, 1, device=q.device)

            for k_kernel, v_proj in zip(k_kernels, v_projected):
                k_h = k_kernel.view(batch_size, self.num_heads, self.head_dim)[:, h]
                v_h = v_proj.view(batch_size, self.num_heads, self.head_dim)[:, h]

                kv_sum += k_h * v_h
                k_sum += k_h.sum(dim=-1, keepdim=True)

            # Normalize and apply query
            normalizer = k_sum + 1e-6
            attn_output_h = q_h * (kv_sum / normalizer) * self.kernel_alpha[h]
            outputs.append(attn_output_h)

        # Concatenate heads
        output = torch.cat(outputs, dim=-1)
        return self.W_o(output)

class ToneDetector:
    """Auto tone detection from text patterns"""

    @staticmethod
    def detect_tone_from_text(text: str) -> str:
        """Detect emotional tone from text"""
        text_lower = text.lower()

        # Sadness indicators
        if any(word in text_lower for word in ["buồn", "khóc", "thương", "tủi"]):
            return "sad"

        # Happiness indicators  
        if any(word in text_lower for word in ["vui", "haha", "cười", "hí"]):
            return "happy"

        # Anger indicators
        if any(word in text_lower for word in ["tức", "giận", "bực", "cáu"]):
            return "angry"

        # Cute/affection indicators
        if any(word in text_lower for word in ["iu", "yêu", "cute", "dễ thương", "nhớ", "thương"]):
            return "cute"

        # Affection indicators
        if any(word in text_lower for word in ["nhớ bạn", "thương bạn", "yêu bạn", "nhớ ghê"]):
            return "affection"

        # Teasing indicators
        if any(word in text_lower for word in ["chọc", "trêu", "đùa", "cà khịa"]):
            return "teasing"

        # Playful indicators
        if any(word in text_lower for word in ["đang làm gì", "cười lên", "vui vẻ"]):
            return "playful"

        # Gentle indicators
        if any(word in text_lower for word in ["thấy", "dễ thương", "nhẹ nhàng"]):
            return "gentle"

        # Comfort indicators
        if any(word in text_lower for word in ["không giỏi", "không sao", "ổn rồi"]):
            return "comfort"

        # Fear indicators
        if any(word in text_lower for word in ["sợ", "lo", "hoang mang"]):
            return "fear"

        return "neutral"
    @staticmethod
    def get_tone_id(tone: str) -> int:
        """Convert tone string to ID"""
        tone_map = {
            "neutral": 0,
            "happy": 1, 
            "sad": 2,
            "angry": 3,
            "fear": 4,
            "cute": 5,
            "affection": 6,
            "teasing": 7,
            "playful": 8,
            "gentle": 9,
            "comfort": 10
        }
        return tone_map.get(tone, 0)

class UltraChronoFireBlock(nn.Module):
    """Enhanced ChronoFire với tất cả improvements"""

    def __init__(self, d_model: int, k_window: int = 16, num_heads: int = 8, 
                 dropout: float = 0.1, use_neuromodulation: bool = True,
                 use_kernelized_attention: bool = True, sparsity_threshold: float = 0.5,
                 layer_idx: int = 0):
        super().__init__()
        self.d_model = d_model
        self.k_window = k_window
        self.num_heads = num_heads
        self.use_neuromodulation = use_neuromodulation
        self.use_kernelized_attention = use_kernelized_attention
        self.sparsity_threshold = sparsity_threshold

        # Multi-head attention với per-head memory
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Per-head memory states
        self.memory_states_per_head = nn.Parameter(torch.zeros(num_heads, self.head_dim))

        # Standard attention components
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Kernelized attention
        if use_kernelized_attention:
            self.kernel_attention = KernelizedAttention(d_model, num_heads)

        # Dilated temporal sampling
        self.dilated_sampler = DilatedTemporalSampler()
        self.temporal_fusion = nn.Linear(d_model * 2, d_model)

        # Neuromodulation
        if use_neuromodulation:
            self.neuromodulation = NeuromodulationUnit(d_model, layer_idx=layer_idx)

        # Enhanced memory mechanism
        self.memory_gate = nn.Linear(d_model * 2, d_model)
        self.memory_update = nn.Linear(d_model, d_model)
        self.W_m = nn.Linear(d_model, d_model)

        # Adaptive thresholding
        self.adaptive_tau = nn.Parameter(torch.tensor(1.0))
        self.tau_gate = nn.Linear(d_model, 1)

        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

        # Feed-forward với sparsity
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        # Sparsity gating
        self.sparsity_gate = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def dilated_temporal_attention(self, x_t: torch.Tensor, x_history: List[torch.Tensor]) -> torch.Tensor:
        """Attention với dilated temporal sampling"""
        if not x_history:
            return self.W_q(x_t)

        # Standard recent window
        recent_window = x_history[-self.k_window//2:] if len(x_history) >= self.k_window//2 else x_history

        # Dilated sampling for far context
        if len(x_history) > self.k_window//2:
            dilated_indices = self.dilated_sampler.get_dilated_indices(
                len(x_history)-1, max_dilation=8, num_samples=self.k_window//2
            )
            dilated_context = [x_history[i] for i in dilated_indices if i < len(x_history)]
        else:
            dilated_context = []

        # Combine recent và dilated context
        all_context = recent_window + dilated_context

        if self.use_kernelized_attention:
            # Use kernel attention for efficiency
            return self.kernel_attention(x_t, all_context, all_context)
        else:
            # Standard multi-head attention
            return self.standard_attention(x_t, all_context)

    def standard_attention(self, x_t: torch.Tensor, context: List[torch.Tensor]) -> torch.Tensor:
        """Standard multi-head attention fallback"""
        if not context:
            return self.W_q(x_t)

        batch_size = x_t.shape[0]
        context_tensor = torch.stack(context, dim=1)  # [batch, seq, d_model]

        Q = self.W_q(x_t).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(context_tensor).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(context_tensor).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)
        return self.W_o(attn_output).squeeze(1)

    def forward(self, x_t: torch.Tensor, states: Dict) -> Tuple[torch.Tensor, Dict]:
        batch_size = x_t.shape[0]

        # Extract states
        m_prev = states.get('memory', torch.zeros(batch_size, self.d_model, device=x_t.device))
        x_history = states.get('history', [])
        emotion_state = states.get('emotion', torch.zeros(batch_size, 8, device=x_t.device))

        # 1. Dilated temporal attention
        h_temporal = self.dilated_temporal_attention(x_t, x_history)
        h_t = F.silu(h_temporal) + x_t
        h_t = self.ln1(h_t)

        # 2. Neuromodulation (if enabled)
        if self.use_neuromodulation:
            m_prev, emotion_state = self.neuromodulation(x_t, emotion_state, m_prev)

        # 3. Enhanced memory mechanism
        dynamic_tau = self.adaptive_tau * torch.sigmoid(self.tau_gate(x_t))
        norm_signal = torch.norm(x_t, dim=-1, keepdim=True)
        memory_trigger = (norm_signal > dynamic_tau).float()

        memory_input = torch.cat([x_t, m_prev], dim=-1)
        memory_gate = torch.sigmoid(self.memory_gate(memory_input))

        m_update = self.memory_update(x_t) * memory_trigger
        m_t = 0.9 * m_prev + 0.1 * (memory_gate * m_update)

        # 4. Memory integration
        memory_contribution = self.W_m(m_t)
        h_t = h_t + memory_contribution
        h_t = self.ln2(h_t)

        # 5. Sparse FFN
        ffn_output = self.ffn(h_t)

        # Sparsity gating - chỉ activate những neurons cần thiết
        sparsity_mask = (torch.sigmoid(self.sparsity_gate(h_t)) > self.sparsity_threshold).float()
        ffn_output = ffn_output * sparsity_mask

        gate_values = torch.sigmoid(self.gate(h_t))
        y_t = h_t + gate_values * ffn_output
        y_t = self.ln3(y_t)

        # Update states
        new_states = {
            'memory': m_t,
            'history': (x_history + [x_t.detach()])[-self.k_window:],  # Keep sliding window
            'emotion': emotion_state,
            'sparsity_stats': sparsity_mask.mean().item()  # For monitoring
        }

        return y_t, new_states

class UltraChronoFireOptimizer:
    """Advanced optimizer cho Ultra ChronoFire"""

    @staticmethod
    def create_optimizer(model, lr: float = 2e-4, weight_decay: float = 0.1) -> torch.optim.Optimizer:
        """Create optimizer với differentiated learning rates"""

        # Phân loại parameters
        temporal_params = []
        memory_params = []
        emotion_params = []
        attention_params = []
        ffn_params = []
        embedding_params = []

        for name, param in model.named_parameters():
            if 'dilated' in name or 'temporal' in name:
                temporal_params.append(param)
            elif 'memory' in name or 'W_m' in name:
                memory_params.append(param)
            elif 'neuromodulation' in name or 'emotion' in name or 'personality' in name:
                emotion_params.append(param)
            elif 'attention' in name or any(x in name for x in ['W_q', 'W_k', 'W_v', 'W_o']):
                attention_params.append(param)
            elif 'ffn' in name:
                ffn_params.append(param)
            elif 'embedding' in name:
                embedding_params.append(param)
            else:
                attention_params.append(param)  # Default group

        # Tạo optimizer với learning rates khác nhau
        param_groups = [
            {'params': temporal_params, 'lr': lr * 1.2, 'weight_decay': weight_decay * 0.8},
            {'params': memory_params, 'lr': lr * 0.8, 'weight_decay': weight_decay * 1.5},
            {'params': emotion_params, 'lr': lr * 0.5, 'weight_decay': weight_decay * 0.5},
            {'params': attention_params, 'lr': lr, 'weight_decay': weight_decay},
            {'params': ffn_params, 'lr': lr * 1.1, 'weight_decay': weight_decay * 1.2},
            {'params': embedding_params, 'lr': lr * 0.6, 'weight_decay': weight_decay * 0.3}
        ]

        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)
        return optimizer

    @staticmethod
    def get_cosine_schedule_with_warmup(optimizer: torch.optim.Optimizer, 
                                      warmup_steps: int = 2000,
                                      total_steps: int = 100000) -> torch.optim.lr_scheduler._LRScheduler:
        """Cosine schedule với warmup cho stable training"""

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)