from gpt4all import GPT4All
import readline # Enable using arrow keys in input box
import pdfplumber
import cupy

def model_name_list(): 
    model_list = GPT4All.list_models()

    for model in model_list:
        print(model["filename"])

# Configuration
DEBUG = False
DEVICE = "cuda:Tesla P40"
N_THREAD = 36
MAX_TOKEN = 4096
N_BATCHS = 128
TOP_K = 40 # 40
TOP_P = 0.4
TEMP = 0.9 # 0.7
REPEAT_PENALTY = 1.18

# Initialization
model_name = "Model Name"
model_path = "Path to Dir Contain Model"
model = GPT4All(model_name=model_name, model_path=model_path, allow_download=False, device=DEVICE, n_threads=N_THREAD, verbose=True)

# Model Information
print("-" * 32 + "Model Information" + "-" * 32)
print(model_name)

# Device Information
print("-" * 32 + "Device Information" + "-" * 32)
print("CUDA Compute Capability : " + cupy.cuda.Device(0).compute_capability)
print("CUDA Memory Info : " + str(cupy.cuda.Device(0).mem_info))
print("\n\n")

file = open('system_prompt.txt', 'r', encoding='utf-8')
system_prompt = file.read()

if DEBUG:
    print(system_prompt)

file.close()

file = open('prompt_template.txt', 'r', encoding='utf-8')
prompt_template = file.read()

if DEBUG:
    print(prompt_template)

file.close()

with model.chat_session(system_prompt=system_prompt, prompt_template=prompt_template):
    while True:
        string = input("> ")
        if not string.find("TXT") == -1:
            with open('./input.txt') as file:
                file_input = file.read()
                string = string.replace("TXT", file_input)
        if not string.find("PDF") == -1:
            try:
                page_input = input("Please input a number for pdf page: ")
                page_num = int(page_input)
            except ValueError:
                print("\nResponds :\n\nShould input a number. Input discarded!\n\nEND\n\n")
                continue
            with pdfplumber.open("input.pdf") as pdf:
                pdf_text = (pdf.pages[page_num]).extract_text()
                string = string.replace("PDF", pdf_text)
        if string == "q":
            break
        response = model.generate(prompt="{}".format(string), max_tokens=MAX_TOKEN, n_batch=N_BATCHS, top_k=TOP_K, top_p=TOP_P, repeat_penalty=REPEAT_PENALTY, temp=TEMP)
        print("\nResponds :\n\n")
        print(response)
        print("\n\nEND\n\n")
