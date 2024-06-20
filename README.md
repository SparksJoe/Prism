<div align="center"><h1>Prism: A Framework for Decoupling and Assessing <br>
the Capabilities of VLMs</h1></div>

![LOGO](/Prism.jpg)

**Prism** is a framework built on [**VLMEvalKit**](https://github.com/open-compass/VLMEvalKit/) for decoupling and accessing the capabilities of **large vision-language models (LVLMs)**. It comprises two distinct stages: 1) **Perception Stage** that first instructs VLMs to extract and express visual information of the image; 2) **Reasoning Stage** that utilizes a external LLM (GPT-3.5, GPT-4, _etc_) to conduct reasoning and answer the question based on the textual information. Prism can both enable the breakdown analysis of VLM capabilities and serve as a solution for vision-language tasks by integrating any given VLM and LLM.

**Demo**
```python
from demo import Perception, Reasoning

text = 'What is this framework about?'
img_path = 'Prism.jpg'

perception_module = Perception(prompt_version='generic', model='GPT4V')
reasoning_module = Reasoning(model='chatgpt-0125')

des = perception_module.generate(text, img_path)
res = reasoning_module.generate(des, text)
```


## Supported Components

- **Perception Module:** ```supported_VLM```(in [VLMEvalkit](https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/config.py)), ```PrismCaptioners```

- **Reasoning Module:** ```gpt models```, ```vllm models```, ```deepseek models```. Check [config](/decouple/config.py) for api calling.


## Preparation

Before running Prism, you need prepare relevant requisites including VLMEvalKit and query-specific instructions. After that, you can check [Usage](https://github.com/SparksJoe/Prism?tab=readme-ov-file#usage) for decoupling VLMs.

### Prepare VLMEvalKit

Check [VLMEvalKit Quickstart](https://github.com/open-compass/VLMEvalKit/blob/main/Quickstart.md) for preparation. Completion of Step 0 and Step 1 is sufficient for Prism. Make sure you are using the environment with VLMEvalKit for Prism.

### Prepare Prism

```bash
git clone https://github.com/SparksJoe/Prism
cd Prism
```

### Prepare Query-Specific Instructions
If you want to use query-specific instructions for perception, use the following command to generate query-specific parts for the required benchmark by reasoning module.
```bash
# Generate query-specific parts
python gen_prompt.py --data MMStar --model chatgpt-0125
```
Make sure that you are using the same reasoning module when generating query-specific parts and conduct reasoning

## Usage

After [Preparation](https://github.com/SparksJoe/Prism?tab=readme-ov-file#preparation), you can run Prism with reasoning modules of ```gpt models``` and ```deepseek models```. For huggingface models like ```llama-70b-chat```, you can deploy them with ```vllm``` following the second part.

### Run a Prism

Use ```run.py``` with ```python``` or ```torchrun``` for Prism. The default setting ```--config``` for Prism is shown in [default config](/config/default_config.json). Here are annotations for arguments.

**Arguments**

- `--data (str, default to 'MMStar')`: Set the benchmark you want to perform Prism on.
- `--model (str, default to 'GPT4V')`: Set the VLM name that is used for the perception module.
- `--infer_model (str, default to 'chatgpt-0125')`: Set the LLM name that is used for the reasoning module.
- `--prompt_version (str, default to 'generic')`: Set the instruction for the perception stage. Check [prompts](/decouple/prompt.py) for details.
- `--mode (str, default to 'all', choices are ['perception', 'reasoning'])`: When `mode` set to "all", Prism will perform perception, reasoning and evaluation; when set to "perception", will only perform perception; when set to "reasoning", will perform perception and reasoning,
- `--nproc (int, default to 4)`: The number of threads for API calling.
- `--postproc (default to False)`: Whether to use random choice for postprocessing.

There are two ways to run your Prism.

**Use a Custom Config.** Write the settings you want into ```self_config.json```.
```bash 
# use python
python run.py --config config/self_config.json

# use torchrun for multi-gpu inference
torchrun --nproc_per_node={gpu_nums} run.py --config config/self_config.json
```
**Use Parameters.** Pass the parameters you modified in the command line, and they will replace the orginial ones in default config.
```bash
# use python
python run.py --model llava_next_yi_34b --infer_model gpt-4-0125

# use torchrun for multi-gpu inference
torchrun --nproc_per_node={gpu_nums} run.py --model llava_next_yi_34b --infer_model gpt-4-0125
```
The command above replaces  ```model``` and ```infer_model``` in the default setting.

**Use Query-Specific Instruction.** You should keep the reasoning module consistent with the prompt version.

```bash
python run.py --model llava_next_yi_34b --prompt_version query-specific_chatgpt-0125 --infer_model chatgpt-0125

# use torchrun for multi-gpu inference
torchrun --nproc_per_node={gpu_nums} run.py --model llava_next_yi_34b --prompt_version query-specific_chatgpt-0125 --infer_model chatgpt-0125
```

**Use PrismCaptioner.** Prism now supports `PrismCaptioner-[2B/7B]`. Just use `--model prismcaptioner-2b`.

### Deploy a HF Model as Reasong Module
You can deploy open-source huggingface models for reasoning with ```vllm```.
First install:
```bash
pip install vllm
```
And then deploy the model with command lines. For `Meta-Llama-3-70B-Instruct`, use
```bash
python -m vllm.entrypoints.openai.api_server \
    -tp {gpu_nums} \
    --model ${MODEL_PATH} \
    --served-model-name llama3-70b-chat \
    --port 8080
```
The default port used in Prism is `8080`, and pay attention to keep `--served-model-name` consistent with model name in [config](/decouple/config.py). Moreover, remember to set stop tokens for vllm models. Then, you can call the model name for reasoning with the command line.
```bash
python run.py --model GPT4V --infer_model llama3-70b-chat 
```

### Other Features
**Merge VLMs.** If you have generated perception results from two different VLMs on the same benchmark with identical prompt, for instance, `GPT4V` and `GeminiProVision`, you can pass `GPT4V~GeminiProVision` to `--model` in order to conduct reasoning on the merged information from them.

**Max Output Length.** For better reasoning performance, you can pass the infer model with a suffix `-2048` like `llama3-70b-chat-2048` to set larger max output length for reasoning module.

## Results Format

The results should be listed as the following structure.

```
└── results
    ├── prompt_version (e.g., generic)
    │   ├── dataset_name (e.g., MMStar)
    │   │   ├── frontend (e.g., GPT4V)
    │   │   │   ├── describe result file
    │   │   │   ├── backend (e.g., chatgpt-0125)
    │   │   │   │   ├── post_infer result files
    │   │   │   │   ├── evaluation result files
    │   │   │   ...
    │   │   └──frontend_backend (e.g., gpt4-0125)
    │   │   ...
    │   └──dataset_name (e.g., MMStar)
    │   ...
    └── prompt_version (e.g., query-specific_chatgpt-0125)
    ...
```

## Acknowledgment
Utmost gratitude to [**Kenny**](https://github.com/kennymckormick)
