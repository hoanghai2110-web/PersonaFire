import torch
import torch.nn.functional as F
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from main.model.custom_tokenizer import UltraChronoFireTokenizer, train_custom_tokenizer
from main.model.model import UltraChronoFireTransformer
from main.model.components import ToneDetector
import json

class UltraChronoFireChat:
    """Interactive chat interface for Ultra ChronoFire"""

    def __init__(self, model_path: str = "./checkpoints/best_model.pt",
                 tokenizer_path: str = "./model/ultra_tokenizer.pkl"):

        # Load custom tokenizer
        if os.path.exists(tokenizer_path):
            self.tokenizer = UltraChronoFireTokenizer.load(tokenizer_path)
        else:
            print("⚠️ Custom tokenizer not found, training new one...")
            self.tokenizer = train_custom_tokenizer("../user_training_data.jsonl")

        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            model_config = checkpoint['model_config']

            self.model = UltraChronoFireTransformer(
                vocab_size=model_config['vocab_size'],
                d_model=model_config['d_model'],
                num_layers=model_config['num_layers'],
                num_heads=model_config['num_heads'],
                k_window=model_config['k_window'],
                personality_mode='fixed'
            ).to(self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model loaded from {model_path}")

        except FileNotFoundError:
            print(f"❌ Model file not found: {model_path}")
            print("Creating new model for demo...")

            self.model = UltraChronoFireTransformer(
                vocab_size=50000,
                d_model=768,
                num_layers=8,
                k_window=32,
                num_heads=12,
                personality_mode='fixed'
            ).to(self.device)

        self.model.eval()

        # Chat state
        self.conversation_history = []
        self.current_personality = 'creative'
        self.model.set_personality(self.current_personality)

    def set_personality(self, personality: str):
        """Change personality"""
        if personality in self.model.personality_profiles:
            self.current_personality = personality
            self.model.set_personality(personality)
            print(f"🎭 Personality changed to: {personality}")
        else:
            available = list(self.model.personality_profiles.keys())
            print(f"❌ Unknown personality. Available: {available}")

    def generate_response(self, user_input: str, max_length: int = 50,
                         temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Generate response to user input"""

        # Detect tone
        detected_tone = ToneDetector.detect_tone_from_text(user_input)
        tone_id = ToneDetector.get_tone_id(detected_tone)

        # Prepare input
        input_ids = self.tokenizer.encode(user_input)
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        # Generate
        with torch.no_grad():
            generated = self.model.generate_response(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                personality=self.current_personality,
                input_text=user_input
            )

        # Decode response
        generated_tokens = generated[0].tolist()
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Store in history
        self.conversation_history.append({
            'user': user_input,
            'assistant': response,
            'tone': detected_tone,
            'personality': self.current_personality
        })

        return response

    def chat_loop(self):
        """Interactive chat loop"""
        print("\n🚀 Ultra ChronoFire Chat Interface")
        print("=" * 50)
        print("Commands:")
        print("  /personality <name> - Change personality")
        print("  /history - Show conversation history")
        print("  /clear - Clear history")
        print("  /quit - Exit")
        print(f"\nCurrent personality: {self.current_personality}")
        print("Available personalities:", list(self.model.personality_profiles.keys()))
        print("\nStart chatting! (type your message)")
        print("-" * 50)

        while True:
            try:
                user_input = input("\n🧑 You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    if user_input == '/quit':
                        print("👋 Goodbye!")
                        break

                    elif user_input == '/history':
                        print("\n📜 Conversation History:")
                        for i, turn in enumerate(self.conversation_history):
                            print(f"  {i+1}. You: {turn['user']}")
                            print(f"     Bot ({turn['personality']}, {turn['tone']}): {turn['assistant']}")
                        continue

                    elif user_input == '/clear':
                        self.conversation_history = []
                        print("🗑️ History cleared!")
                        continue

                    elif user_input.startswith('/personality '):
                        personality = user_input.split(' ', 1)[1]
                        self.set_personality(personality)
                        continue

                    else:
                        print("❌ Unknown command")
                        continue

                # Generate response
                print("\n🤖 Thinking...", end='', flush=True)
                response = self.generate_response(user_input)

                # Display with personality info
                detected_tone = ToneDetector.detect_tone_from_text(user_input)
                print(f"\r🤖 Bot ({self.current_personality}, detected: {detected_tone}): {response}")

            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue

    def save_conversation(self, filepath: str = "conversation_log.json"):
        """Save conversation history"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        print(f"💾 Conversation saved to {filepath}")

def main():
    """Main chat function"""

    # Try to load model, fall back to demo mode
    try:
        chat = UltraChronoFireChat("./checkpoints/best_model.pt")
    except:
        print("⚠️ No trained model found, using demo mode with random weights")
        chat = UltraChronoFireChat("")

    # Start chat
    chat.chat_loop()

    # Save conversation
    if chat.conversation_history:
        chat.save_conversation()

if __name__ == "__main__":
    main()