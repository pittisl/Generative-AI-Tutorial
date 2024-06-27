<h1><img align="center" height="40" src="genairoadmap_logo.jpg"> Generative AI Roadmap</h1>

> A subjective learning guide for generative AI research including curated list of articles and projects

Generative AI is a hot topic today :fire: and this roadmap is designed to help beginners quickly gain basic knowledge and find useful resources of Generative AI. Even experts are welcome to refer to this roadmap to recall old knowledge and develop new ideas.

## Table of Content
- [Background Knowledge](#background-knowledge)
  - [Neural Networks Inference and Training](#neural-networks-inference-and-training)
  - [Transformer Architecture](#transformer-architecture)
  - [Common Transformer-based Models](#common-transformer-based-models)
  - [Miscellaneous](#miscellaneous)
- [Large Language Models (LLMs)](#large-language-models-llms)
  - [Pretraining and Fine-tuning](#pretraining-and-finetuning)
  - [Prompting](#prompting)
  - [Evaluation](#evaluation)
  - [Dealing with Long Context](#dealing-with-long-context)
  - [Efficient Fine-tuning](#efficient-finetuning)
  - [Model Merging](#model-merging)
  - [Efficient Generation](#efficient-generation)
  - [Knowledge Editing](#knowledge-editing)
  - [LLM-powered Agents](#llm-powered-agents)
  - [Findings](#findings)
  - [Open Challenges](#open-challenges)
- [Diffusion Models](#diffusion-models)
  - [Image Generation](#image-generation)
  - [Video Generation](#video-generation)
  - [Audio Generation](#audio-generation)
  - [Pretraining and Fine-tuning](#pretraining-and-finetuning-1)
  - [Evaluation](#evaluation-1)
  - [Efficient Generation](#efficient-generation-1)
  - [Knowledge Editing](#knowledge-editing-1)
  - [Open Challenges](#open-challenges-1)
- [Large Multimodal Models (LMMs)](#large-multimodal-models-lmms)
  - [Model Architectures](#model-architectures)
  - [Towards Embodied Agents](#towards-embodied-agents)
  - [Open Challenges](#open-challenges-2)
- [Beyond Transformers](#beyond-transformers)
  - [Implicitly Structured Parameters](#implictly-structured-parameters)
  - [New Model Architectures](#new-model-architectures)


## Background Knowledge
This section should help you learn or regain the basic knowledge of neural networks (e.g., backpropagation), get you familiar with the transformer architecture, and describe some common transformer-based models.

### Neural Networks Inference and Training

Are you very familiar with the following classic neural network structures?
- [Multi-Layer Perceptron (MLP)](https://www.tensorflow.org/guide/core/mlp_core)
- [Convolutional Neural Network (CNN)](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)
- [Recurrent Neural Network (RNN)](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)

:pencil: If so, you should be able to answer these questions:
- Why do CNNs work better than MLPs on images?
- Why do RNNs work better than MLPs on time-series data?
- What's the difference between GRU and LSTM?

Backpropagation (BP) is the base of NN training. **You will not be an AI expert if you don't understand BP**. There are many textbooks and online tutorials teaching BP, but unfortunately, most of them don't present formulas in vectorized/tensorized forms. The BP formula of an NN layer is indeed as neat as its forward pass formula. This is exactly how BP is implemented and should be implemented. To understand BP, please read the following materials:

- [Neural Networks and Deep Learning](http://ndl.ethernet.edu.et/bitstream/123456789/88552/1/2018_Book_NeuralNetworksAndDeepLearning.pdf) [Chapter 3.2 especially 3.2.6]
- [meProp: Sparsified Back Propagation for Accelerated Deep Learning with Reduced Overfitting (ICML 2017)](https://arxiv.org/pdf/1706.06197.pdf) [Section 2.1]
- [Resprop: Reuse sparsified backpropagation (CVPR 2020)](http://openaccess.thecvf.com/content_CVPR_2020/papers/Goli_ReSprop_Reuse_Sparsified_Backpropagation_CVPR_2020_paper.pdf) [Section 3.1]

:pencil: If you understand BP, you should be able to answer these questions:
- How will you describe the BP of a convolutional layer?
- What is the ratio of the computing cost (i.e., number of floating point operations) between forward pass and backward pass of a dense layer?
- How will you describe the BP of an MLP with two dense layers sharing the same weight matrix?

### Transformer Architecture
Transformer is the base architecture of existing large generative models. It's necessary to understand every component in the transformer. Please read the following materials:
- [Attention Is All You Need (NeurIPS 2017)](https://arxiv.org/pdf/1706.03762.pdf) [Original Paper]
- [An image is worth 16x16 words: Transformers for image recognition at scale (ICLR 2021)](https://arxiv.org/pdf/2010.11929.pdf) [Vision Transformer]
- [Neural machine translation with a Transformer and Keras](https://www.tensorflow.org/text/tutorials/transformer) [Great Explanation for Multihead Attention]
- [FLOPs of a Transformer Block](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Pan_Scalable_Vision_Transformers_ICCV_2021_supplemental.pdf) [Let's practice calculating FLOPs]
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/pdf/1911.02150v1.pdf) [Multi-Query Attention]
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245v3) [Grouped-Query Attention]
- [Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864.pdf) [Understand Positional Embedding]
- [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/) [Understand Positional Embedding]
- [Teacher Forcing vs Scheduled Sampling vs Normal Mode](https://rentruewang.github.io/learning-machine/layers/transformer/training/teacher/teacher.html) [Teacher Forcing in Transformer Training]
- [FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU](https://arxiv.org/pdf/2303.06865) [See section 3 - generative inference to learn how LLMs peform generation based on KV cache]
- [Contextual Position Encoding: Learning to Count What’s Important](https://arxiv.org/pdf/2405.18719) [Context-dependent positional encoding]


:pencil: If you understand transformers, you should be able to answer these questions:
- What are the pros and cons of tranformers compared to RNNs (simultaneously attending, training parallelism, complexity)?
- Can you caculate the FLOPs of Grouped-Query Attention? When does it degrade MultiHead Attention and Multi-Query Attention?
- What does the causal attention mask look like and why?
- How will you describe the training of decoder-only transformers step by step?
- Why is RoPE better than sinusoidal positional encoding?

### Common Transformer-based Models
- [Learning transferable visual models from natural language supervision](http://proceedings.mlr.press/v139/radford21a/radford21a.pdf) [CLIP]
- [Emerging Properties in Self-Supervised Vision Transformers (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf) [DINO]
- [Masked autoencoders are scalable vision learners (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf) [MAE]
- [Scaling Vision with Sparse Mixture of Experts (NeurIPS 2021)](https://arxiv.org/pdf/2106.05974.pdf) [MoE]
- [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/pdf/2404.02258) [MoD]

### Miscellaneous

* [Einsum is easy and useful](https://ejenner.com/post/einsum/) [A great tutorial for using einsum/einops]
* [Open-Endedness is Essential for Artificial Superhuman Intelligence (ICML 2024)](https://arxiv.org/pdf/2406.04268) [Thoughts on achieving superhuman AI]

## Large Language Models (LLMs)
LLMs are transformers. They can be categorized into encoder-only, encoder-decoder, and decoder-only architectures, as shown in the LLM evolutionary tree below [[image source]](https://github.com/Mooler0410/LLMsPracticalGuide/blob/main/imgs/tree.jpg). Check [milestone papers](https://github.com/Hannibal046/Awesome-LLM?tab=readme-ov-file#milestone-papers) of LLMs.

![LLM Evolutionary Tree](https://github.com/Mooler0410/LLMsPracticalGuide/raw/main/imgs/tree.jpg)

Encoder-only model can be used to extract sentence features but lacks generative power. Encoder-decoder and decoder-only models are used for text generation. In particular, most existing LLMs prefer decoder-only structures due to stronger repesentational power. Intuitively, encoder-decoder models can be considered a sparse version of decoder-only models and the information decays more from encoder to decoder. Check this [paper](https://arxiv.org/pdf/2304.04052.pdf) for more details.

### Pretraining and Finetuning
LLMs are typically pretrained from trillions of text tokens by model publishers to internalize the natural language structure. Today's model developers also conduct instructional fine-tuning and Reinforcement Learning from Human Feedback (RLHF) to teach the model to follow human instructions and generate answers aligned with human preference. The users can then download the published model and finetune it on small personal datasets (e.g., movie dialog). Due to huge amount of data, pretraining requires massive computing resources (e.g., more than thousands of GPUs) which is unaffordable by individuals. On the other hand, fine-tuning is less resource-hungry and can be done with a few GPUs. 

The following materials can help you understand the pretraining and fine-tuning process:

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805) [Pretraining and Finetuning of Encoder-only LLMs]
- [Scaling Instruction-Finetuned Language Models](https://arxiv.org/pdf/2210.11416) [Pretraining and Instructional Finetuning]
- [Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf)
- [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165) [Decoder-only LLMs] [[中文导读 by 李沐]](https://www.bilibili.com/video/BV1AF411b7xQ)

More tutorials can be found [here](https://github.com/Hannibal046/Awesome-LLM?tab=readme-ov-file#tutorials).

### Prompting
Prompting techniques for LLMs involve crafting input text in a way that guides the model to generate desired responses or outputs. Here are the useful resources to help you write better prompts:
- [[DAIR.AI] Prompt Engineering Guide](https://github.com/dair-ai/Prompt-Engineering-Guide)
- [Awesome ChatGPT Prompts](https://github.com/f/awesome-chatgpt-prompts) - A collection of prompt examples to be used with the ChatGPT model
- [Awesome Deliberative Prompting](https://github.com/logikon-ai/awesome-deliberative-prompting) - How to ask LLMs to produce reliable reasoning and make reason-responsive decisions
- [AutoPrompt](https://github.com/ucinlp/autoprompt) - An automated method based on gradient-guided search to create prompts for a diverse set of NLP tasks.

### Evaluation
Evaluation tools for large language models help assess their performance, capabilities, and limitations across different tasks and datasets. Here are some common evaluation strategies:
- **Automatic Evaluation Metrics**: These metrics assess model performance automatically without human intervention. Common metrics include:

  - [BLEU](https://huggingface.co/spaces/evaluate-metric/bleu): Measures the similarity between generated text and reference text based on n-gram overlap.
  - [ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge): Evaluates text summarization by comparing overlapping n-grams between generated and reference summaries.
  - [Perplexity](https://huggingface.co/docs/transformers/en/perplexity): Measures how well a language model predicts a sample of text. Lower perplexity indicates better performance. It is equivalent to the exponentiation of the cross-entropy between the data and model predictions.
  - [F1 Score](https://huggingface.co/spaces/evaluate-metric/f1): Measures the balance between precision and recall in tasks like text classification or named entity recognition.

- **Human Evaluation**: Human judgment is essential for assessing the quality of generated text comprehensively. Common human evaluation methods include:

  - **Human Ratings**: Human annotators rate generated text based on criteria such as fluency, coherence, relevance, and grammaticality.
  - **Crowdsourcing Platforms**: Platforms like Amazon Mechanical Turk or Figure Eight facilitate large-scale human evaluation by crowdsourcing annotations.
  - **Expert Evaluation**: Domain experts assess model outputs to gauge their suitability for specific applications or tasks.

- **Benchmark Datasets**: Standardized datasets enable fair comparison of models across different tasks and domains. Examples include:

  - [TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension](https://github.com/mandarjoshi90/triviaqa)
  - [HellaSwag: Can a Machine Really Finish Your Sentence?](https://aclanthology.org/P19-1472.pdf)
  - [GSM8K: Training Verifiers to Solve Math Word Problems](https://github.com/openai/grade-school-math)
  - A complete list can be found [here](https://github.com/SihyeongPark/Awesome-LLM-Benchmark?tab=readme-ov-file#llm-datasetsbenchmarks-list)

- Model Analysis Tools: Tools for analyzing model behavior and performance include:

  - [Automated Interpretability](https://github.com/openai/automated-interpretability) - Code for automatically generating, simulating, and scoring explanations of neuron behavior
  - [LLM Visualization](https://bbycroft.net/llm) - Visualizing LLMs in low level.
  - [Attention Analysis](https://github.com/clarkkev/attention-analysis) - Analyzing attention maps from BERT transformer.
  - [Neuron Viewer](https://openaipublic.blob.core.windows.net/neuron-explainer/neuron-viewer/index.html) - Tool for viewing neuron activations and explanations.

A complete list can be found [here](https://github.com/JShollaj/awesome-llm-interpretability?tab=readme-ov-file#llm-interpretability-tools)

Standard evaluation frameworks for existing LLMs include:
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - A framework for few-shot evaluation of language models.
- [lighteval](https://github.com/huggingface/lighteval) - a lightweight LLM evaluation suite that Hugging Face has been using internally.
- [OLMO-eval](https://github.com/allenai/OLMo-Eval) - a repository for evaluating open language models.
- [instruct-eval](https://github.com/declare-lab/instruct-eval) - This repository contains code to quantitatively evaluate instruction-tuned models such as Alpaca and Flan-T5 on held-out tasks.


### Dealing with Long Context

Dealing with long contexts poses a challenge for large language models due to limitations in memory and processing capacity. Existing techniques include:
- Efficient Transformers
  - [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)
  - [Reformer: The efficient transformer (ICLR 2020)](https://arxiv.org/abs/2001.04451)
- State Space Models
  - [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention (ICML 2020)](https://arxiv.org/abs/2006.16236)
  - [Rethinking attention with performers](https://arxiv.org/abs/2009.14794)
- Length Extrapolation
  - [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
  - [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
  - [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)
- Long Term Memory
  - [MemoryBank: Enhancing Large Language Models with Long-Term Memory](https://arxiv.org/abs/2305.10250)
  - [Unleashing Infinite-Length Input Capacity for Large-scale Language Models with Self-Controlled Memory System](https://arxiv.org/abs/2304.13343)

A complete list can be found [here](https://github.com/Xnhyacinth/Awesome-LLM-Long-Context-Modeling?tab=readme-ov-file#-papers)


### Efficient Finetuning

Parameter-Efficient Fine-Tuning (PEFT) methods enable efficient adaptation of large pretrained models to various downstream applications by only fine-tuning a small number of (extra) model parameters instead of all the model's parameters:
  - Prompt Tuning: [The power of scale for parameter-efficient prompt tuning](https://arxiv.org/pdf/2104.08691.pdf?trk=public_post_comment-text)
  - Prefix Tuning: [Prefix-tuning: Optimizing continuous prompts for generation](https://arxiv.org/pdf/2101.00190.pdf%EF%BC%89)
  - LoRA: [Lora: Low-rank adaptation of large language models](https://arxiv.org/pdf/2106.09685)
  - [Towards a Unified View of Parameter-Efficient Transfer Learning](https://arxiv.org/pdf/2110.04366)
  - [LoRA Learns Less and Forgets Less](https://arxiv.org/pdf/2405.09673)

More work can be found in [Huggingface PEFT paper collection](https://huggingface.co/collections/PEFT/peft-papers-6573a1a95da75f987fb873ad) and it's highly recommended to practice with [HuggingFace PEFT API](https://github.com/huggingface/peft).

### Model Merging
Model merging refers to merging two or more LLMs trained on different tasks into a single LLM. This technique aims to leverage the strengths and knowledge of different models to create a more robust and capable model. For example, a LLM for code generation and another LLM for math prolem solving can be merged together so that the merged model is capable of doing both code generation and math problem solving.

The model merging is intriguing because it can be effectively achieved with very simple and cheap algorithms (e.g., linear combination of model weights). Here are some representative papers and reading materials:

* [Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://arxiv.org/abs/2203.05482)
* [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089)
* [Merge Large Language Models with mergekit](https://huggingface.co/blog/mlabonne/merge-models)

More papers about model merging can be found [here](https://huggingface.co/collections/osanseviero/model-merging-65097893623330a3a51ead66)


### Efficient Generation
Accelerating decoding of LLMs is crucial for improving inference speed and efficiency, especially in real-time or latency-sensitive applications. Here are some representative work of speeding up decoding process of LLMs:

- [Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time (ICML 2023 Oral)](https://openreview.net/forum?id=wIPIhHd00i)
- [LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models (EMNLP 2023)](https://arxiv.org/abs/2310.05736)
- [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)
- [SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification](https://arxiv.org/abs/2305.09781)
- [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
- [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/pdf/2404.19737)
- [Layer Skip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/pdf/2404.16710)

More work about accelerating LLM decoding can be found via [Link 1](https://github.com/horseee/Awesome-Efficient-LLM?tab=readme-ov-file#inference-acceleration) and [Link 2](https://github.com/DefTruth/Awesome-LLM-Inference).

### Knowledge Editing
Knowledge editing aims to efficiently modify LLMs behaviors, such as reducing bias and revising learned correlations. It includes many topics such as knowledge localization and unlearning. Representative work includes:
- [Memory-Based Model Editing at Scale (ICML 2022)](https://arxiv.org/abs/2206.06520)
- [Transformer-Patcher: One Mistake worth One Neuron (ICLR 2023)](https://arxiv.org/abs/2301.09785)
- [Massive Editing for Large Language Model via Meta Learning (ICLR 2024)](https://arxiv.org/pdf/2311.04661.pdf)
- [A Unified Framework for Model Editing](https://arxiv.org/abs/2403.14236)
- [Transformer Feed-Forward Layers Are Key-Value Memories (EMNLP 2021)](https://arxiv.org/abs/2012.14913)
- [Mass-Editing Memory in a Transformer](https://arxiv.org/abs/2210.07229)

More papers can be found [here](https://github.com/zjunlp/KnowledgeEditingPapers).


### LLM-powered Agents
By receiving massive training, LLMs digest world knowledge and are able to follow input instructions precisely. With these amazing capabilities, LLMs can play as agents that are possible to autonomously (and collaboratively) solve complex tasks, or simulate human interactions. Here are some representative papers of LLM agents:

* [Generative Agents: Interactive Simulacra of Human Behavior (UIST 2023)](https://arxiv.org/pdf/2304.03442) [LLMs simulate human society in video games]
* [SOTOPIA: Interactive Evaluation for Social Intelligence in Language Agents (ICLR 2024)](https://arxiv.org/abs/2310.11667) [LLMs simulate social interactions]
* [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/pdf/2305.16291) [LLMs live in the Minecraft world]
* [Large Language Models as Tool Makers (ICLR 2024)](https://arxiv.org/abs/2305.17126) [LLMs create their own reusable tools (e.g., in python functions) for problem-solving]
* [MetaGPT: Meta Programming for Multi-Agent Collaborative Framework](https://arxiv.org/pdf/2308.00352) [LLMs as a team for automated software development]
* [WebArena: A Realistic Web Environment for Building Autonomous Agents (ICLR 2024)](https://arxiv.org/abs/2307.13854) [LLMs use web applications]
* [Mobile-Env: An Evaluation Platform and Benchmark for LLM-GUI Interaction](https://arxiv.org/pdf/2305.08144) [LLMs use mobile applications]
* [HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face (NeurIPS 2023)](https://arxiv.org/pdf/2303.17580) [LLMs seek models in huggingface for problem-solving]
* [AGENTGYM: Evolving Large Language Model-based
Agents across Diverse Environments](https://arxiv.org/pdf/2406.04151) [Diverse interactive environments and tasks for LLM-based agents]

A complete list of papers, platforms, and evaluation tools can be found [here](https://github.com/hyp1231/awesome-llm-powered-agent?tab=readme-ov-file#general-reasoning--planning--tool-using).

### Findings

- [Your Transformer is Secretly Linear](https://arxiv.org/pdf/2405.12250)
- [Not All Language Model Features Are Linear](https://arxiv.org/abs/2405.14860)

### Open Challenges
LLMs face several open challenges that researchers and developers are actively working to address. These challenges include:
- Hallucination
  - [A Comprehensive Survey of Hallucination Mitigation Techniques in Large Language Models](https://arxiv.org/pdf/2401.01313.pdf)
- Model Compression
  - [A Comprehensive Survey of Compression Algorithms for Language Models](https://arxiv.org/pdf/2401.15347.pdf)
- Evaluation
  - [Evaluating Large Language Models: A Comprehensive Survey](https://arxiv.org/pdf/2310.19736.pdf)
- Reasoning
  - [A Survey of Reasoning with Foundation Models](https://arxiv.org/pdf/2312.11562.pdf)
- Explainability
  - [From Understanding to Utilization: A Survey on Explainability for Large Language Models](https://arxiv.org/pdf/2401.12874.pdf)
- Fairness
  - [A Survey on Fairness in Large Language Models](https://arxiv.org/abs/2308.10149)
- Factuality
  - [A Survey on Factuality in Large Language Models: Knowledge, Retrieval and Domain-Specificity](https://arxiv.org/abs/2310.07521)
- Knowledge Integration
  - [Trends in Integration of Knowledge and Large Language Models: A Survey and Taxonomy of Methods, Benchmarks, and Applications](https://arxiv.org/pdf/2311.05876.pdf)

A complete list can be found [here](https://github.com/HqWu-HITCS/Awesome-LLM-Survey?tab=readme-ov-file#challenge-of-llm).

## Diffusion Models
Diffusion models aim to approxmiate the probability distribution of a given data domain, and provide a way to generate samples from its approximated distribution. Their goals are similar to other popular generative models, such as [VAE](https://arxiv.org/pdf/1606.05908), [GANs](https://arxiv.org/pdf/1406.2661), and [Normalizing Flows](https://arxiv.org/pdf/1908.09257).

The working flow of diffusion models is featured with two process:
1. **Forward process (diffusion process):** it progressively applies noise to the original input data step by step until the data completely becomes noise.
2. **Reverse process (denoising process):** an NN model (e.g., CNN or tranformer) is trained to estimate the noise being applied in each step during the forward process. This trained NN model can then be used to generate data from noise input. Existing diffusion models can also accept other signals (e.g., text prompts from users) to condition the data generation.

Check [this awesome blog](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) and more introductory tutorials can be found [here](https://github.com/diff-usion/Awesome-Diffusion-Models#introductory-posts). Diffusion models can be used to generate images, audios, videos, and more, and there are many subfields related to diffusion models as shown below [[image source]](https://github.com/YangLing0818/Diffusion-Models-Papers-Survey-Taxonomy):

![Diffusion Model Taxonomy](https://user-images.githubusercontent.com/62683396/227244860-3608bf02-b2af-4c00-8e87-6221a59a4c42.png)

### Image Generation
Here are some representative papers of diffusion models for image generation:
- [High-Resolution Image Synthesis with Latent Diffusion Models (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html)
- [Palette: Image-to-image diffusion models (SIGGRAPH 2022)](https://arxiv.org/pdf/2111.05826)
- [Image Super-Resolution via Iterative Refinement](https://arxiv.org/pdf/2104.07636)
- [Inpainting using Denoising Diffusion Probabilistic Models (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Lugmayr_RePaint_Inpainting_Using_Denoising_Diffusion_Probabilistic_Models_CVPR_2022_paper.html)
- [Adding Conditional Control to Text-to-Image Diffusion Models (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.pdf)

More papers can be found [here](https://github.com/YangLing0818/Diffusion-Models-Papers-Survey-Taxonomy/tree/main?tab=readme-ov-file#1-computer-vision).

### Video Generation
Here are some representative papers of diffusion models for video generation:
- [Video Diffusion Models](https://arxiv.org/pdf/2204.03458)
- [Flexible Diffusion Modeling of Long Videos (NeurIPS 2022)](https://arxiv.org/pdf/2205.11495)
- [Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/pdf/2311.15127)
- [I2VGen-XL: High-Quality Image-to-Video Synthesis via Cascaded Diffusion Models](https://arxiv.org/pdf/2311.04145.pdf)

More papers can be found [here](https://github.com/wangkai930418/awesome-diffusion-categorized?tab=readme-ov-file#video-generation).

### Audio Generation
Here are some representative papers of diffusion models for audio generation:
- [Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech](https://proceedings.mlr.press/v139/popov21a.html)
- [Text-to-Audio Generation using Instruction-Tuned LLM and Latent Diffusion Model](https://arxiv.org/pdf/2304.13731v1.pdf)
- [Zero-Shot Voice Conditioning for Denoising Diffusion TTS Models](https://arxiv.org/abs/2206.02246)
- [EdiTTS: Score-based Editing for Controllable Text-to-Speech](https://arxiv.org/abs/2110.02584)
- [ProDiff: Progressive Fast Diffusion Model For High-Quality Text-to-Speech](https://arxiv.org/abs/2207.06389)

More papers can be found [here](https://diff-usion.github.io/Awesome-Diffusion-Models/#text-to-speech).

### Pretraining and Finetuning
Similar to other large generative models, diffusion models are also pretrained on large amount of web data (e.g., [LAION-5B dataset](https://proceedings.neurips.cc/paper_files/paper/2022/file/a1859debfb3b59d094f3504d5ebb6c25-Paper-Datasets_and_Benchmarks.pdf)) and consume massive computing resources. Users can download the released weights can further fine-tune the model on personal datasets.

Here are some representative papers of efficient fine-tuning of diffusion models:
- [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Ruiz_DreamBooth_Fine_Tuning_Text-to-Image_Diffusion_Models_for_Subject-Driven_Generation_CVPR_2023_paper.html)
- [An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion (ICLR 2023)](https://openreview.net/forum?id=NAQvF08TcyG)
- [Custom Diffusion: Multi-Concept Customization of Text-to-Image Diffusion (cvpr 2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Kumari_Multi-Concept_Customization_of_Text-to-Image_Diffusion_CVPR_2023_paper.html)
- [Controlling Text-to-Image Diffusion by Orthogonal Finetuning (NeurIPS 2023)](https://nips.cc/virtual/2023/poster/72033)

More papers can be found [here](https://github.com/wangkai930418/awesome-diffusion-categorized?tab=readme-ov-file#new-concept-learning).

It's highly recommended to do some practice with [Huggingface Diffusers API](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image).

### Evaluation
Here we talk about evaluation of diffusion models for image generation. Many existing image quality metrics can be applied.
- [CLIP score](https://arxiv.org/abs/2104.08718): CLIP score measures the compatibility of image-caption pairs. Higher CLIP scores imply higher compatibility. CLIP score was found to have high correlation with human judgement.
- [Fréchet Inception Distance (FID)](https://arxiv.org/abs/1706.08500): FID aims to measure how similar are two datasets of images. It is calculated by computing the Fréchet distance between two Gaussians fitted to feature representations of the Inception network
- [CLIP directional similarity](https://arxiv.org/abs/2108.00946): It measures the consistency of the change between the two images (in CLIP space) with the change between the two image captions.

More image quality metrics and calculation tools can be found [here](https://github.com/chaofengc/IQA-PyTorch/blob/main/docs/ModelCard.md).

### Efficient Generation
Diffusion models require multiple forward steps over to generate data, which is expensive. Here are some representative papers of diffusion models for efficient generation:

- [Gotta Go Fast When Generating Data with Score-Based Models](https://arxiv.org/abs/2105.14080)
- [Fast Sampling of Diffusion Models with Exponential Integrator](https://arxiv.org/abs/2204.13902)
- [Learning fast samplers for diffusion models by differentiating through sample quality](https://openreview.net/forum?id=VFBjuF8HEp)
- [Accelerating Diffusion Models via Early Stop of the Diffusion Process](https://arxiv.org/abs/2205.12524)

More papers can be found [here](https://github.com/YangLing0818/Diffusion-Models-Papers-Survey-Taxonomy/tree/main?tab=readme-ov-file#1-efficient-sampling).

### Knowledge Editing
Here are some representative papers of knowledge editing for diffusion models:
- [Erasing Concepts from Diffusion Models (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Gandikota_Erasing_Concepts_from_Diffusion_Models_ICCV_2023_paper.html)
- [Editing Massive Concepts in Text-to-Image Diffusion Models](https://arxiv.org/abs/2403.13807)
- [Forget-Me-Not: Learning to Forget in Text-to-Image Diffusion Models](https://arxiv.org/abs/2303.17591)

More papers can be found [here](https://github.com/wangkai930418/awesome-diffusion-categorized?tab=readme-ov-file#remove-concept).

### Open Challenges
Here are some survey papers talking about the challenges faced by diffusion models.
- [A Survey of Diffusion Based Image Generation Models](https://arxiv.org/pdf/2308.13142)
- [A Survey on Video Diffusion Models](https://arxiv.org/pdf/2310.10647)
- [State of the Art on Diffusion Models for Visual Computing](https://arxiv.org/abs/2310.07204)
- [Diffusion Models in NLP: A Survey](https://arxiv.org/abs/2303.07576)

## Large Multimodal Models (LMMs)
Typical LMMs are constructed by connecting and fine-tuning existing pretrained unimodal models. Some are also pretrained from scratch. Check how LMMs evolve in the image below [[image source]](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models).

![Diffusion Model Taxonomy](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/raw/main/images/timeline.jpg)

### Model Architectures
There are many different ways of contructing LMMs. Representative architectures include:
- [Language Models are General-Purpose Interfaces](https://arxiv.org/pdf/2206.06336.pdf)
- [Flamingo: A Visual Language Model for Few-Shot Learning (NeurIPS 2022)](https://arxiv.org/abs/2204.14198)
- [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation (ICML 2022)](https://arxiv.org/abs/2201.12086)
- [BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models (ICML 2023)](https://arxiv.org/pdf/2301.12597.pdf)
- [mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration](https://arxiv.org/abs/2311.04257)
- [Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks](https://arxiv.org/pdf/2311.06242)
- [Dense Connector for MLLMs](https://arxiv.org/abs/2405.13800)

More papers can be found via [Link 1](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models?tab=readme-ov-file#foundation-models) and [Link 2](https://github.com/junkunyuan/Awesome-Large-Multimodal-Models?tab=readme-ov-file#foundation-models).

### Towards Embodied Agents
By combining LMMs with robots, researchers aim to develop AI systems that can perceive, reason about, and act upon the world in a more natural and intuitive way, with potential applications spanning robotics, virtual assistants, autonomous vehicles, and beyond. Here are some representative work of realizing embodied AI with LMMs:

- [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/pdf/2212.06817)
- [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/pdf/2307.15818)
- [RT-H: Action Hierarchies Using Language](https://arxiv.org/pdf/2403.01823)
- [PaLM-E: An Embodied Multimodal Language Model](https://arxiv.org/pdf/2303.03378)
- [TRANSIC: Sim-to-Real Policy Transfer by Learning from Online Correction](https://arxiv.org/pdf/2405.10315)

More papers can be found via [Link 1](https://github.com/ChanganVR/awesome-embodied-vision) and [Link 2](https://github.com/haoranD/Awesome-Embodied-AI).

Here are some popular simulators and datasets to evaluate LMMs performance for embodied AI:
- [Habitat 3.0: An Embodied AI simulation platform for studying collaborative human-robot interaction tasks in home environments](https://aihabitat.org/habitat3/)
- [ProcTHOR-10K: 10K Interactive Household Environments for Embodied AI](https://procthor.allenai.org/)
- [ARNOLD: A Benchmark for Language-Grounded Task Learning With Continuous States in Realistic 3D Scenes](https://arnold-benchmark.github.io/)
- [LEGENT: Open Platform for Embodied Agents](https://arxiv.org/pdf/2404.18243)
- [RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots](https://arxiv.org/pdf/2406.02523)

More resources can be found [here](https://github.com/ChanganVR/awesome-embodied-vision?tab=readme-ov-file#-simulators).

### Open Challenges
Here are some survey papers talking about open challenges for LMM-enabled embodied AI:
- [The Rise and Potential of Large Language Model Based Agents: A Survey](https://arxiv.org/pdf/2309.07864)
- [Vision-Language Navigation with Embodied Intelligence: A Survey](https://arxiv.org/abs/2402.14304)
- [A Survey of Embodied AI: From Simulators to Research Tasks](https://arxiv.org/pdf/2103.04918.pdf)
- [A Survey on LLM-based Autonomous Agents](https://github.com/Paitesanshi/LLM-Agent-Survey)
- [Mindstorms in Natural Language-Based Societies of Mind](https://arxiv.org/pdf/2305.17066.pdf)

## Beyond Transformers
Researchers are trying to explore new models other than transformers. The efforts include implicitly structuring model parameters and defining new model architectures.

### Implictly Structured Parameters

- [Monarch Mixer: Revisiting BERT, Without Attention or MLPs](https://arxiv.org/pdf/2310.12109)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/pdf/2312.00752)

### New Model Architectures

  - [Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/pdf/2302.10866)
  - [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/pdf/2305.13048)
  - [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/pdf/2307.08621)
  - [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/pdf/2312.00752)
  - [KAN:Kolmogorov–Arnold Networks](https://arxiv.org/pdf/2404.19756)
  - [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/pdf/2405.21060)

Here is an awesome tutorial for [state space models](https://srush.github.io/annotated-s4/).












