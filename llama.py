import torch
from transformers import pipeline, LlamaForCausalLM, LlamaTokenizerFast

model_id = "unsloth/Llama-3.2-1B-Instruct"

# Specify the path to your local model directory
model_path = "./models/Llama-3.2-1B-Instruct"

# Load the tokenizer and model from the local path
model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16, # if args.device == "cuda" else torch.float32,
            device_map="auto"
        )
tokenizer = LlamaTokenizerFast.from_pretrained(model_path, truncation_side="left")


# Set up the pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)

# Define the initial system prompt
system_prompt = "You are a consultant for a bank, whose job is to give the bank suggestion. Given the analysis on loan reason and credit risk, you are tasked with giving report on the assessment of this loans and credit risk in short sentence(at most 100 words))"

# Template for messages
def get_response_from_llama(user_input, loan_reason, loan_status):
    prompt = f"{system_prompt}\nThe loan reason is: {loan_reason}. The predict loan status is: {loan_status}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt")
    response = pipe(prompt, max_length=256, do_sample=True, temperature=0.7)
    # Extract the assistant's reply
    assistant_reply = response[0]['generated_text'].split("Assistant:")[-1].strip()
    return assistant_reply
