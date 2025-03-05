import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2-large"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)



# Set a pad token that is different from the eos token
tokenizer.pad_token = tokenizer.eos_token
# Set the model to evaluation mode
model.eval()




def remove_repeated_sentences(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    unique_sentences = []
    for sentence in sentences:
        if sentence not in unique_sentences:
            unique_sentences.append(sentence)
        else:
            break  # Stop adding sentences after the first repetition
    return ' '.join(unique_sentences)

def clean_response(response_text, prompt):
    # Remove the prompt from the response
    stripped_response = response_text.replace(prompt, '').strip()

    # Split the stripped response text into lines
    lines = stripped_response.split('\n')

    combined_lines = " ".join(line.strip() for line in lines if line.strip())
    return remove_repeated_sentences(combined_lines)


def generate_response(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    
    # Calculate attention mask
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    # Generate response
    with torch.no_grad():
        output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id,attention_mask=attention_mask)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    cleaned_response = clean_response(response, prompt)

    return cleaned_response


print("Chatbot: Hi there! How can I help you?")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    response = generate_response(user_input)
    print("Chatbot:", response)