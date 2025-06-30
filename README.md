# 🇻🇳 UltraChronoFire - 50M tham số

UltraChronoFire là mô hình ngôn ngữ cảm xúc tiếng Việt, tối ưu cho hội thoại tự nhiên, có cảm xúc và logic im lặng.

## ⚙️ Cấu hình
- 16 layers · 1024 dim · 16 heads
- 50K steps · Custom tokenizer
- Checkpoint resume & incremental training

## 🌟 Điểm nổi bật

- ✨ Neuromodulation: điều chỉnh trí nhớ theo cảm xúc
- 🔄 Kernel Attention: attention nhanh hơn, O(n)
- 🔥 Emotion-Gated Memory: nhớ lâu khi có cảm xúc mạnh
- 🧠 Tokenizer riêng cho tiếng Việt (BPE)
- 💾 Nhẹ (~45M tham số), chạy tốt trên CPU/GPU yếu

## 🚀 Cách dùng
```bash
git clone https://github.com/hoanghai2110-web/PersonaFire
cd PersonaFire/model

# Huấn luyện
python train.py

# Sau khi train xong
python generate_chat.py
