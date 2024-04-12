# Generative AI Roadmap
> A subjective learning guide for generative AI research including curated list of articles and projects

Generative AI is a hot topic today :fire: and this roadmap is designed to help beginners quickly gain basic knowledge and skills of Generative AI. Even experts are welcome to refer to this roadmap as a checklist to develop new ideas.

:bulb: Tips: A good habit of learning is to ask yourself questions.

## Contents
- [Background Knowledge](#background-knowledge)
  - [Neural Networks Inference and Training](#neural-networks-inference-and-training)
  - [Transformer Architecture](#transformer-architecture)
  - [Common Transformer-based Models](#common-transformer-based-models)

- [Large Language Models (LLMs)](#large-language-models-llms)
  - [Pretraining and Fine-tuning](#pretraining-and-finetuning)
  - Prompting
  - Evaluation
  - Dealing with Long Context
  - Efficient Fine-tuning
  - Efficient Generation
  - Open Challenges

- Diffusion Models
  - Image Generation
  - Video Generation
  - Audio Generation
  - Pretraining and Fine-tuning
  - Evaluation
  - Efficient Generation
  - Open Challenges

- Large Multimodal Models
  - Model Architecture
  - Towards Embodied Agents
  - Open Challenges

- New Model Architectures
  - Hyena
  - RWKV
  - RetNet
  - Mamba

- Explainability

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

Backpropagation (BP) is the base of NN training. --You will not be an AI expert if you don't understand BP--. There are many textbooks and online tutorials teaching BP, but unfortunately, most of them don't present formulas in vectorized/tensorized forms. The BP formula of an NN layer is indeed as neat as its forward pass formula. This is exactly how BP is implemented and should be implemented. To understand BP, please read the following materials:

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
- [Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864.pdf) [Understand Positional Embedding]
- [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/) [Understand Positional Embedding]
- [Teacher Forcing vs Scheduled Sampling vs Normal Mode](https://rentruewang.github.io/learning-machine/layers/transformer/training/teacher/teacher.html) [Teacher Forcing in Transformer Training]

:pencil: If you understand transformers, you should be able to answer these questions:
- What are the pros and cons of tranformers compared to RNNs?
- How does the causal attention mask look like and why?
- How will you describe the training of decoder-only transformers step by step?
- Why is RoPE better than sinusoidal positional encoding in the original transformer paper?

### Common Transformer-based Models
- [Learning transferable visual models from natural language supervision](http://proceedings.mlr.press/v139/radford21a/radford21a.pdf) [CLIP]
- [Emerging Properties in Self-Supervised Vision Transformers (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf) [DINO]
- [Masked autoencoders are scalable vision learners (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf) [MAE]
- [Scaling Vision with Sparse Mixture of Experts (NeurIPS 2021)](https://arxiv.org/pdf/2106.05974.pdf) [MoE]
- [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/pdf/2404.02258) [MoD]

## Large Language Models (LLMs)
LLMs are transformers. They can be categorized into encoder-only, encoder-decoder, and decoder-only architectures, as shown in the LLM evolutionary tree below [[image source]](https://github.com/Mooler0410/LLMsPracticalGuide/blob/main/imgs/tree.jpg). Encoder-only model can be used to extract sentence features but lacks generative power. Encoder-decoder and decoder-only models are used for text generation. In particular, most existing LLMs prefer decoder-only structures due to stronger repesentational power. Intuitively, encoder-decoder models can be considered a sparse version of decoder-only models and the information decays more from encoder to decoder. Check this [paper](https://arxiv.org/pdf/2304.04052.pdf) for more details.


![LLM Evolutionary Tree](https://github.com/Mooler0410/LLMsPracticalGuide/blob/main/imgs/tree.jpg)

### Pretraining and Finetuning
LLMs are typically pretrained from trillions of text tokens by model publishers to internalize the natural language structure. Today's model developers also conduct instructional fine-tuning and Reinforcement Learning from Human Feedback (RLHF) to teach the model to follow human instructions and generate answers aligned with human preference.

The users can then download the published model and finetune it on small personal datasets (e.g., movie dialog). Due to huge amount of data, pretraining requires massive computing resources (e.g., more than thousands of GPUs) which is unaffordable by individuals. On the other hand, fine-tuning is less resource-hungry and can be done with a few GPUs. 

The following materials can help you understand the pretraining and fine-tuning process:

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805) [Pretraining and Finetuning of Encoder-only LLMs]
- [Scaling Instruction-Finetuned Language Models](https://arxiv.org/pdf/2210.11416) [Pretraining and Instructional Finetuning]
- [Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf)
- [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165) [Decoder-only LLMs] [[中文导读 by 李沐]](https://www.bilibili.com/video/BV1AF411b7xQ/?spm_id_from=333.999.0.0)

More tutorials can be found [here](https://github.com/Hannibal046/Awesome-LLM?tab=readme-ov-file#tutorials).

### Prompting



