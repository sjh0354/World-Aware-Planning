
# World-aware Planning Narratives for Large Vision-Language Models

## 🔥Overview
This repository contains the official implementation of our paper on enhancing large vision-language models (LVLMs) with world-aware planning narratives. Our approach bridges the gap between high-level task instructions and nuanced real-world environments by integrating contextual world knowledge into planning systems [1].

## 🚀Key Features
- Framework for enhancing vision-language models with contextual reasoning
- Significant improvements in planning ability, include commonsense reasoning and long-horizon planning tasks
- Outperforms proprietary systems like GPT-4o and Claude-3.5-Sonnet 

## 🖥️ Installation
Note: we follow the installation process of EB-ALFRED from EmbodiedBench

Many thanks to EmbodiedBench for a comprehensive benchmark! (https://github.com/EmbodiedBench/EmbodiedBench)

**Download repo**
```bash
git clone git@github.com:EmbodiedBench/EmbodiedBench.git
cd World-Aware-Planning
```

**Environment for EB-ALFRED**
```bash
cd EmbodiedBench
conda env create -f conda_envs/environment.yaml 
conda activate embench
pip install -e .
```

**Start Headless Server**

Start tmux in a new tmux terminal
```bash
Xvfb :1 -screen 0 1024x768x24 &
```

**Download dataset from huggingface.**

```bash
cd EmbodiedBench
conda activate embench
git clone https://huggingface.co/datasets/EmbodiedBench/EB-ALFRED
mv EB-ALFRED embodiedbench/envs/eb_alfred/data/json_2.1.0
```

Run the following code to ensure the EB-ALFRED environment is working correctly. Remember to start headless server.

```
conda activate embench
python -m embodiedbench.envs.eb_alfred.EBAlfEnv
```

## 🚀 Quick Start
**Testing models with vllm server**
```python
## Step 1, start a vllm server (suggest with tmux)
python -m vllm.entrypoints.openai.api_server --model=your_model

## Step 2, running the evaluation
conda activate embench
export remote_url='IP_address:port/v1' # set the address for access, e.g., http://localhost:8000.
python -m embodiedbench.main env=eb-hab model_name=your_model_name exp_name='baseline' 
```

## 🔧 Model Settings
Our framework employs Qwen2.5-VL-72B-Instruct as the teacher model for instruction augmentation and reasoning generation. We evaluate our approach on two foundation model series:
- Qwen2.5-VL (Qwen2.5-VL-7B-Instruct)
- InternVL3 (InternVL3-8B) 

## 🚀Performance
Our approach achieves substantial improvements over baseline methods:
- +60.7 absolute improvement in average task success rates with Qwen2.5-VL
- +60.0 in commonsense reasoning
- +70.0 in long-horizon planning 

The enhanced open-source models outperform recent proprietary systems by a large margin.

## 🛠️Evaluation
We evaluate on the EB-ALFRED benchmark from EmbodiedBench, using Success Rate (SR) as the primary metric.

## Citation
If you find this work useful for your research, please cite our paper:
```bibtex
@misc{shi2025worldawareplanningnarrativesenhance,
      title={World-aware Planning Narratives Enhance Large Vision-Language Model Planner}, 
      author={Junhao Shi and Zhaoye Fei and Siyin Wang and Qipeng Guo and Jingjing Gong and Xipeng Qiu},
      year={2025},
      eprint={2506.21230},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21230}, 
}
```
