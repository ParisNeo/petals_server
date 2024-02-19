from fastapi import FastAPI, BaseModel
import petals
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM

import subprocess
from ascii_colors import ASCIIColors, trace_exception
from typing import Optional
import threading
app = FastAPI()



class Infos:
    cancel_gen:bool=False,
    tokenizer:AutoTokenizer=None,
    model:AutoDistributedModelForCausalLM=None
    model_device=None

class Config:
    model_name:str


model_infos = Infos()
config = Config()
def build_model():
            model_infos.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            model_infos.model = AutoDistributedModelForCausalLM.from_pretrained(config.model_name,
                                                          device_map="auto")
            model_infos.model_device = model_infos.model.parameters().__next__().device


def start_server(model_name, node_name, device):
    command = [
        "python3",
        "-m",
        "petals.cli.run_server",
        model_name,
        "--public_name",
        node_name,
        "--device",
        device,
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as ex:
        trace_exception(ex)

# ----------------------------------- Generation -----------------------------------------
class LollmsGenerateRequest(BaseModel):
    text: str
    model_name: Optional[str] = None
    personality: Optional[int] = None
    n_predict: Optional[int] = 1024
    stream: bool = False
    temperature: float = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repeat_penalty: Optional[float] = None
    repeat_last_n: Optional[int] = None
    seed: Optional[int] = None
    n_threads: Optional[int] = None

@app.post("/generate")
async def lollms_generate(request: LollmsGenerateRequest):
    """ Endpoint for generating text from prompts using the LoLLMs fastAPI server.

    Args:
    Data model for the Generate Request.
    Attributes:
    - text: str : representing the input text prompt for text generation.
    - model_name: Optional[str] = None : The name of the model to be used (it should be one of the current models)
    - personality_id: Optional[int] = None : The name of the mounted personality to be used (if a personality is None, the endpoint will just return a completion text). To get the list of mounted personalities, just use /list_mounted_personalities
    - n_predict: int representing the number of predictions to generate.
    - stream: bool indicating whether to stream the generated text or not.
    - temperature: float representing the temperature parameter for text generation.
    - top_k: int representing the top_k parameter for text generation.
    - top_p: float representing the top_p parameter for text generation.
    - repeat_penalty: float representing the repeat_penalty parameter for text generation.
    - repeat_last_n: int representing the repeat_last_n parameter for text generation.
    - seed: int representing the seed for text generation.
    - n_threads: int representing the number of threads for text generation.

    Returns:
    - If the elf_server binding is not None:
    - If stream is True, returns a StreamingResponse of generated text chunks.
    - If stream is False, returns the generated text as a string.
    - If the elf_server binding is None, returns None.
    """

    try:
        text = request.text
        n_predict = request.n_predict
        stream = request.stream
        if stream:

            output = {"text":"","waiting":True,"new":[]}
            def generate_chunks():
                lk = threading.Lock()

                def callback(chunk):
                    if infos["cancel_gen"]:
                        return False
                    if chunk is None:
                        return
                    output["text"] += chunk
                    # Yield each chunk of data
                    lk.acquire()
                    try:
                        antiprompt = detect_antiprompt(output["text"])
                        if antiprompt:
                            ASCIIColors.warning(f"\nDetected hallucination with antiprompt: {antiprompt}")
                            output["text"] = remove_text_from_string(output["text"],antiprompt)
                            lk.release()
                            return False
                        else:
                            output["new"].append(chunk)
                            lk.release()
                            return True
                    except Exception as ex:
                        trace_exception(ex)
                        lk.release()
                        return True
                def chunks_builder():
                    elf_server.binding.generate(
                                            text, 
                                            n_predict, 
                                            callback=callback, 
                                            temperature=request.temperature if request.temperature is not None else elf_server.config.temperature,
                                            top_k=request.top_k if request.top_k is not None else elf_server.config.top_k, 
                                            top_p=request.top_p if request.top_p is not None else elf_server.config.top_p,
                                            repeat_penalty=request.repeat_penalty if request.repeat_penalty is not None else elf_server.config.repeat_penalty,
                                            repeat_last_n=request.repeat_last_n if request.repeat_last_n is not None else elf_server.config.repeat_last_n,
                                            seed=request.seed if request.seed is not None else elf_server.config.seed,
                                            n_threads=request.n_threads if request.n_threads is not None else elf_server.config.n_threads
                                        )
                    output["waiting"] = False
                thread = threading.Thread(target=chunks_builder)
                thread.start()
                current_index = 0
                while (output["waiting"] and elf_server.cancel_gen == False):
                    while (output["waiting"] and len(output["new"])==0):
                        time.sleep(0.001)
                    lk.acquire()
                    for i in range(len(output["new"])):
                        current_index += 1                        
                        yield output["new"][i]
                    output["new"]=[]
                    lk.release()
                elf_server.cancel_gen = False

            return StreamingResponse(iter(generate_chunks()))
        else:
            output = {"text":""}
            def callback(chunk, chunk_type:MSG_TYPE=MSG_TYPE.MSG_TYPE_CHUNK):
                # Yield each chunk of data
                output["text"] += chunk
                antiprompt = detect_antiprompt(output["text"])
                if antiprompt:
                    ASCIIColors.warning(f"\nDetected hallucination with antiprompt: {antiprompt}")
                    output["text"] = remove_text_from_string(output["text"],antiprompt)
                    return False
                else:
                    return True
            elf_server.binding.generate(
                                            text, 
                                            n_predict, 
                                            callback=callback,
                                            temperature=request.temperature if request.temperature is not None else elf_server.config.temperature,
                                            top_k=request.top_k if request.top_k is not None else elf_server.config.top_k, 
                                            top_p=request.top_p if request.top_p is not None else elf_server.config.top_p,
                                            repeat_penalty=request.repeat_penalty if request.repeat_penalty is not None else elf_server.config.repeat_penalty,
                                            repeat_last_n=request.repeat_last_n if request.repeat_last_n is not None else elf_server.config.repeat_last_n,
                                            seed=request.seed if request.seed is not None else elf_server.config.seed,
                                            n_threads=request.n_threads if request.n_threads is not None else elf_server.config.n_threads
                                        )
            return output["text"]
    except Exception as ex:
        trace_exception(ex)
        elf_server.error(ex)
        return {"status":False,"error":str(ex)}



@app.get("/")
def read_root():
    return {"Hello": "World"}


def main():
    parser = argparse.ArgumentParser(description="Start the server with the given parameters.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--node_name", type=str, required=True, help="Name of the node")
    parser.add_argument("--device", type=str, required=True, help="Device to run the model on")
    parser.add_argument("--host", type=str, default="localhost", help="Host for the server")
    parser.add_argument("--port", type=int, default=8064, help="Port for the server")

    args = parser.parse_args()

    start_server(args.model_name, args.node_name, args.device)
    
    uvicorn.run("app:app", host=args.host, port=args.port)

if __name__ == "__main__":
    main()
