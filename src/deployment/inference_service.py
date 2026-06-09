import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class RAGInferenceGateway:
    """
    Loads the trained, consolidated model weights and exposes an interface
    for real-time RAG security analysis queries.
    """
    def __init__(self, model_dir="./saved_cybersec_model"):
        self.model_dir = model_dir
        
        # Fall back to base model if you haven't run a local training run yet
        if not os.path.exists(model_dir):
            print(f"⚠️  Optimized weights not found at {model_dir}. Loading baseline model...")
            self.model_dir = "EleutherAI/pythia-70m"
            
        print(f"📦 Loading optimized model architecture from {self.model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir)
        
        # Move to GPU if available for sub-millisecond production inference
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval() # Set to evaluation mode (turns off dropout/batchnorm)

    def analyze_security_prompt(self, context: str, query: str) -> str:
        """
        Combines retrieved RAG context with a user query to generate a secure response.
        """
        # Construct a clean system prompt combining context + query
        structured_prompt = (
            f"Context: {context}\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        
        inputs = self.tokenizer(structured_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad(): # Disable gradient tracking to save massive memory during inference
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        decoded_response = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        # Strip out the prompt text to return just the clean answer
        return decoded_response[len(structured_prompt):].strip()

if __name__ == "__main__":
    # Local verification test
    gateway = RAGInferenceGateway()
    sample_context = "Firewall Rule 14 explicitly blocks inbound traffic on port 22 except from IP 192.168.1.50."
    sample_query = "Can IP 192.168.1.99 connect via SSH?"
    
    print("\n🔍 Running local gateway inference test...")
    response = gateway.analyze_security_prompt(sample_context, sample_query)
    print(f"🤖 Model Security Response: {response}")