# Sprouts-Jittor

新芽式的考核（新芽训练营、大组保研）均需采用 Jittor 复现代码。本 Repo 收集整理一些目前已有的 Jittor 开源代码，从而可以方便正在接受考核的同学尽量避免重复实现已有 Jittor 代码的工作。需要注意的是，本 Repo 收集的并不全面，对于具体工作，还请更大范围搜索后再判定是否适合作为考核选题。

**Under Construction**



更新日期：2025.8.10



## 基础网络架构

| Year | Venue | Model            | Document Title                                               | 代码地址                                   | 事由     |
| ---- | ----- | ---------------- | ------------------------------------------------------------ | ------------------------------------------ | -------- |
| 2021 | ICCV  | Swin Transformer | Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows | https://github.com/Miaehal/jt_pytorch_swin | 保研面试 |



## 持续学习

| Year | Venue | Model   | Document Title                                               | 代码地址                                      | 事由     |
| ---- | ----- | ------- | ------------------------------------------------------------ | --------------------------------------------- | -------- |
| 2016 | ECCV  | LWF     | Learning Without Forgetting                                  | https://github.com/kira9339/jittor_lwf        | 新芽培育 |
| 2018 | CVPR  | PackNet | PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning | <https://github.com/Fun-James/PackNet-Jittor> | 新芽培育 |
|      |       |         |                                                              |                                               |          |
|      |       |         |                                                              |                                               |          |



## 图像重建与复原

| Year | Venue | Model         | Document Title                                               | 代码地址                                                | 事由     |
| ---- | ----- | ------------- | ------------------------------------------------------------ | ------------------------------------------------------- | -------- |
| 2021 | CVPR  | LIIF          | Learning Continuous Image Representation with Local Implicit Image Function | https://github.com/a1ei/liif_jittor                     | 保研面试 |
| 2021 | CVPR  | LIIF          | Learning Continuous Image Representation with Local Implicit Image Function | https://github.com/WindATree/LIIF-Jittor                | 保研面试 |
| 2022 | CVPR  | Blind2Unblind | Blind2Unblind: Self-Supervised Image Denoising with Visible Blind Spots | https://github.com/Qu-a-n/B2UwithJittor                 | 保研面试 |
| 2024 | TPAMI | Flow_PWC      | Self-Supervised Deep Blind Video Super-Resolution            | https://github.com/Athena-Re/Self-Blind-VSR-with-Jittor | 保研面试 |



## 通用目标检测

| Year | Venue   | Model        | Document Title                                               | 代码地址                                            | 事由     |
| ---- | ------- | ------------ | ------------------------------------------------------------ | --------------------------------------------------- | -------- |
| 2015 | NeurIPS | Faster R-CNN | Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks | https://github.com/arianna-h/exam_jittor            | 保研面试 |
| 2015 | NeurIPS | Faster R-CNN | Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks | https://github.com/wwiinndd/TwoStageDeteciton       | 保研面试 |
| 2015 | NeurIPS | Faster R-CNN | Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks | https://github.com/Holysmiles/faster_rcnn_jittor    | 保研面试 |
| 2017 | CVPR    | RetinaNet    | Focal Loss for Dense Object Detection                        | https://github.com/Running-Turtle1/jittor-retinanet | 保研面试 |
| 2021 | CVPR    | GFL V2       | Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection | https://github.com/FuZhongyuan/GFocalV2             | 保研面试 |
| 2020 | ECCV    | DETR         | End-to-End Object Detection with Transformers                | https://github.com/Ber0ton/DETR-Jittor-and-Pytorch/ | 保研面试 |



## 小样本学习（分类、分割、检测）

| Year | Venue          | Model      | Document Title                                               | 代码地址                                                     | 事由     |
| ---- | -------------- | ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| 2015 | MICCAI         | UNet       | U-Net: Convolutional Networks for Biomedical Image Segmentation | https://github.com/xyt732/xyt                                | 保研面试 |
| 2023 | ArXiv          | FS-MedSAM2 | RFS-MedSAM2: Exploring the Potential of SAM2 for Few-Shot Medical Image Segmentation without Fine-tuning | https://github.com/MasterAlphZero/JtToFSMedSAM2              | 保研面试 |
| 2022 | CVPR           | Imagen     | Prompt-to-Prompt Image Editing with Cross-Attention Control  | https://github.com/NJUST-wyx/20---prompt-to-prompt?tab=readme-ov-file | 保研面试 |
| 2023 | IEEE Xplore    | CPANet     | Cross Position Aggregation Network for Few-shot Strip Steel Surface Defect Segmentation | https://github.com/ZSLsherly/CPANet-Pytorch-Jittor/tree/master/CPANet | 保研面试 |
| 2020 | Neurocomputing | ResNet-50  | Revisiting Metric Learning for Few-Shot Image Classification | https://github.com/withernova/Revisiting_Metric_Learning_for_Few-Shot_Image_Classification/tree/ori_essay | 保研面试 |
| 2023 | MIDL           | UNet       | MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model | https://github.com/jarambler/MedSegDiff-Jittor/tree/main     | 保研面试 |
| 2024 | PMLR           | MedSegDiff | MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model | https://github.com/XD-XXF/MedSegDiff-Jittor                  | 保研面试 |
| 2023 | ICCV           | SCCAN      | Self-Calibrated Cross Attention Network for Few-Shot Segmentation | https://github.com/AllenReder/SCCAN-jittor                   | 保研面试 |
| 2020 | IEEE           | CPANet     | Cross Position Aggregation Network for Few-Shot Strip Steel Surface Defect Segmentation | https://github.com/SidneyLu/StarGAN2j                        | 保研面试 |
|      |                |            |                                                              |                                                              |          |
|      |                |            |                                                              |                                                              |          |
|      |                |            |                                                              |                                                              |          |
|      |                |            |                                                              |                                                              |          |
|      |                |            |                                                              |                                                              |          |
|      |                |            |                                                              |                                                              |          |
|      |                |            |                                                              |                                                              |          |
|      |                |            |                                                              |                                                              |          |





## 恶劣环境视觉感知

| Year | Venue | Model        | Document Title                                               | 代码地址                                          | 事由     |
| ---- | ----- | ------------ | ------------------------------------------------------------ | ------------------------------------------------- | -------- |
| 2021 | ICCV  | DC-ShadowNet | DC-ShadowNet: Single-Image Hard and Soft Shadow Removal Using Unsupervised Domain-Classifier Guided Network | https://github.com/Wang-Yi-Yu/Jittor_DC-ShadowNet | 新芽培育 |
|      |       |              |                                                              |                                                   |          |



## 图像生成模型

| Year | Venue | Model      | Document Title                                           | 代码地址                              | 事由     |
| ---- | ----- | ---------- | -------------------------------------------------------- | ------------------------------------- | -------- |
| 2020 | CVPR  | StarGAN v2 | StarGAN v2: Diverse Image Synthesis for Multiple Domains | https://github.com/SidneyLu/StarGAN2j | 新芽培育 |
|      |       |            |                                                          |                                       |          |



## 大语言模型

| Year | Venue | Model | Document Title                                               | 代码地址                                       | 事由     |
| ---- | ----- | ----- | ------------------------------------------------------------ | ---------------------------------------------- | -------- |
| 2023 | ICLR  | ReAct | ReAct**:** Synergizing Reasoning and Acting in Language Models | https://github.com/August-Liu2004/ReAct-Jittor | 保研面试 |
|      |       |       |                                                              |                                                |          |
|      |       |       |                                                              |                                                |          |
|      |       |       |                                                              |                                                |          |





## 持续学习：从基础到前沿

| Year | Venue   | Model      | Document Title                                           | 代码地址                              | 事由       |
| ---- | -----   | ---------- | -------------------------------------------------------- | ------------------------------------- | ---------- |
| 2025 | CVPR | MoAL | Knowledge Memorization and Rumination for Pre-trained Model-based Class-Incremental Learning | https://github.com/thirwave/MoalJt | 保研面试 |
| 2022 | CVPR                                  | l2p          | Prompt-based Continual Learning Official Jax Implementation  | https://github.com/chillybreeze-hao/l2p_jittor               | 保研面试 |                                        |      |
| 2025 | CVPR  | ResNet-50 | Order-Robust Class Incremental Learning: Graph-Driven Dynamic Similarity Grouping | https://github.com/Sitaye/GDDSG-Jittor | 保研面试 |
| 2023 | NeurIPS | RanPAC | RanPAC: Random Projections and Pre-trained Models for Continual Learning | https://github.com/Fluoroantimonic-H/RanPAC-jittor | 保研面试 |
| 2022 | CVPR | L2P | Learning to Prompt for Continual Learning | https://github.com/paraliine/l2p-jittor | 保研面试 |

## 扩散式生成模型：从噪声到图像

| Year | Venue   | Model      | Document Title                                           | 代码地址                              | 事由       |
| ---- | -----   | ---------- | -------------------------------------------------------- | ------------------------------------- | ---------- |
| 2020 | NIPS | DDPM | Denoising Diffusion Probabilistic Models | https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models | 保研面试 |
| 2020 | NeurIPS                               | DDPM         | Denoising Diffusion Probabilistic Models                     | https://github.com/linyuxin666666/LYX_DDPM                   | 保研面试 |
| 2021 | ICLR                                  | DDIM         | Denoising Diffusion Implicit Models                          | https://github.com/AprilS6/ddpm_ddim_with_jittor_pytorch     | 保研面试 |
| 2021 | ICLR                                  | DDIM         | Denoising Diffusion Implicit Models                          | https://github.com/hupeach/DDIM                              | 保研面试 |
| 2025 | ICLR  | DiT-S/2 | One Step Diffusion via Shortcut Models                       | https://github.com/yoshimatsuu/shortcut-models-jittor | 保研面试 |
| 2022 | CVPR  | LDMs    | High-Resolution Image Synthesis with Latent Diffusion Models | https://github.com/Hakurei-Reimu-Gensokyo/simple-ldm  | 保研面试 |
| 2020 | NeurIPS | DDPM | Denoising Diffusion Probabilistic Models | https://github.com/renren1988/jittor_diffusion?tab=readme-ov-file | 保研面试 |
| 2020 | ICLR | DDIM | Denoising Diffusion Implicit Models | https://github.com/MoonKirito/Jittor-DDIM | 保研面试 |
| 2020 | IEEE  | DDPM | Denoising Diffusion Probabilistic Models | https://github.com/zyxiang2004/JitNoise | 保研面试 |
| 2022 | CVPR | LDM | High-Resolution Image Synthesis with Latent Diffusion Models | https://github.com/zhuhe3331/Jittor-LDM-MNIST | 保研面试 |
| 2022 | NeurIPS | CFG | Classifier-Free Diffusion Guidance | https://github.com/Sunnet314159/CFG-Jittor |保研面试  |
| 2020 | ICLR | DCN | Diffusion Generative Models: From Noise to Image | https://github.com/x-y20/jittor-dcn | 保研面试 |
## 洞见隐微：红外弱小目标检测

| Year | Venue   | Model      | Document Title                                           | 代码地址                              | 事由       |
| ---- | -----   | ---------- | -------------------------------------------------------- | ------------------------------------- | ---------- |
| 2024 | TGRS | L2SKNet | Saliency at the Helm: Steering Infrared Small Target Detection with Learnable Kernels| https://github.com/Yejiaxuan/L2SKNet-Jittor | 保研面试 |
| 2023 | IEEE Transactions on Image Processing | DNANet       | Dense Nested Attention Network for Infrared Small Target Detection | https://github.com/cicdean/Jittor-Implementation-of-DNANet   | 保研面试 |
| 2024 | CVPR  | MSHNet | Infrared Small Target Detection with Scale and Location Sensitivity | https://github.com/Hareplace/MSHNet_jittor/tree/master | 保研面试 |
| 2022 | TIP | UIUNet | UIU-Net: U-Net in U-Net for Infrared Small Object Detection        | https://github.com/V-Ane/UIUNet | 保研面试 |
| 2022 | TIP | DNANet | Dense nested attention network for infrared small target detection | https://github.com/Sauruang/DNANet-Jittor-Implementation/tree/main | 保研面试 |

## 发生在成像之前：RAW 数据噪声建模与去噪

| Year | Venue   | Model      | Document Title                                           | 代码地址                              | 事由       |
| ---- | -----   | ---------- | -------------------------------------------------------- | ------------------------------------- | ---------- |
| 2023 | CVPR | DNF | DNF: Decouple and Feedback Network for Seeing in the Dark| https://github.com/RRRRReus/DNF-Jittor-Reproduction | 保研面试 |

 ## 事件相机：多模态融合与目标检测

| Year | Venue   | Model      | Document Title                                           | 代码地址                              | 事由       |
| ---- | -----   | ---------- | -------------------------------------------------------- | ------------------------------------- | ---------- |
| 2023 | AAAI                                  | DMANet       | Dual Memory Aggregation Network for Event-based Object Detection with Learnable Representation | https://github.com/DayDayupupu/DMANet_jittor                 | 保研面试 |
| 2023 | AAAI  | DMANet | Dual Memory Aggregation Network for Event-Based Object Detection with Learnable Representation | https://github.com/Yuyciciccc/DMANet-Jittor | 保研面试 |

 ## 视觉语言大模型：模型加速与高效微调

| Year | Venue   | Model      | Document Title                                           | 代码地址                              | 事由       |
| ---- | -----   | ---------- | -------------------------------------------------------- | ------------------------------------- | ---------- |
| 2022 | ICLR                                  | LoRA         | LoRA: Low-Rank Adaptation of Large Language Models           | https://github.com/zhenrys/LoRA-GPT2-E2E-pytorch-jittor/tree/master/by-jittor/NLG/src | 保研面试 |
| 2019 | Open Ai 博客 | GPT2 | Language Models are Unsupervised Multitask Learners | https://github.com/GsjResilient/lora_jittor | 保研面试 |
| 2023 | CVPR | EfficientViT | EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention | https://github.com/Scarlett19uyu/Efficientvitcls-jittor | 保研面试 |
| 2023 | ICLR | MoA | Vision-Language Large Models: Model Acceleration and Efficient Fine-tuning | https://github.com/MorningYin/MoA_Jittor | 保研面试 |

 ## 轻量化与高效部署：深度学习模型的实用之旅

| Year | Venue   | Model      | Document Title                                           | 代码地址                              | 事由       |
| ---- | -----   | ---------- | -------------------------------------------------------- | ------------------------------------- | ---------- |
| 2022 | NeurIPS                               | DKD          | Decomposed Knowledge Distillation for Class-Incremental Semantic Segmentation | https://github.com/xp-nb/Jittor_DKD                          | 保研面试 |
| 2021 | CVPR | RepVGG | RepVGG: Making VGG-style ConvNets Great Again | https://github.com/NeikuiColacat/RepVGG-Jittor-CIFAR100 | 保研面试 |
| 2022 | CVPR | DKD | Lightweight and Efficient Deployment: A Practical Journey of Deep Learning Models | https://github.com/Envy6163/DKD-Jittor.git | 保研面试 |

 ## 面向下游任务的多模态图像融合

| Year | Venue   | Model      | Document Title                                           | 代码地址                              | 事由       |
| ---- | -----   | ---------- | -------------------------------------------------------- | ------------------------------------- | ---------- |
| 2023 | Information Fusion | PSFusion | Rethinking the necessity of image fusion in high-level vision tasks: A practical infrared and visible image fusion network based on progressive semantic injection and scene fidelity | https://github.com/yzbcs/PSFusionJittor | 保研面试 |
| 2025 | CVPR | SelfGIFNet | One Model for ALL: Low-Level Task Interaction Is a Key to Task-Agnostic Image Fusion | https://github.com/Kallen6669/SelfGIFNet?tab=readme-ov-file | 保研面试 |

 ## 大语言模型Agent

| Year | Venue   | Model      | Document Title                                           | 代码地址                              | 事由       |
| ---- | -----   | ---------- | -------------------------------------------------------- | ------------------------------------- | ---------- |
| 2023 | arXiv | Transformer | LLaMA: Open and Efficient Foundation Language Models | https://github.com/Hanrong-li/llama_jittor/tree/master | 保研面试 |
| 2022 | ICLR  | GPT2_LoRA          | LoRA: Low-Rank Adaptation of Large Language Models           | https://github.com/thereflection84/GPT2_LoRA_Jittor | 保研面试 |
| 2019 | arxiv | LoRA on DistilBERT | DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter | https://github.com/github-haoyan/github-haoyan      | 保研面试 |
| 2022 | ICLR | LoRA | LoRA: Low-Rank Adaptation of Large Language Models | https://github.com/waywooKwong/LoRA-Jittor | 保研面试 |
| 2022 | ICLR | LoRA | LoRA: Low-Rank Adaptation of Large Language Models | https://github.com/Estella999/LoRA-jittor | 保研面试 |
| 2023 | ICLR | ReAct | ReAct: Synergizing Reasoning and Acting in Language Models | https://github.com/peichenxi77/React?tab=readme-ov-file | 保研面试 |

 ## 伪装目标检测：从方法到认知

| Year | Venue   | Model      | Document Title                                           | 代码地址                              | 事由       |
| ---- | -----   | ---------- | -------------------------------------------------------- | ------------------------------------- | ---------- |
| 2021 | CVPR  | PFNet   | Camouflaged Object Segmentation With Distraction Mining      | https://github.com/ejekess/PFNet_jittor       | 保研面试 |
| 2020 | CVPR | SINet | Camouflaged Object Detection: From Method to Cognition | https://github.com/WUKEYINGING/Jittor-SINet-WKY | 保研面试 |


 ## 遥感感知模型

| Year | Venue   | Model      | Document Title                                           | 代码地址                              | 事由       |
| ---- | -----   | ---------- | -------------------------------------------------------- | ------------------------------------- | ---------- |
| 2023 | ICCV | JDet-Nanoputian | Large Selective Kernel Network for Remote Sensing Object Detection | https://github.com/eptq00/JDet-Nanoputian | 保研面试 |

 ## 多模态图像生成

| Year | Venue   | Model      | Document Title                                           | 代码地址                              | 事由       |
| ---- | -----   | ---------- | -------------------------------------------------------- | ------------------------------------- | ---------- |
| 2020 | CVPR | StarGAN v2 | Multi-modal Image Generation | https://github.com/fwp101/stargan-v2-jittor | 保研面试 |
| 2018 | CVPR | StarGAN | Multi-modal Image Generation | https://github.com/lkhup/StarGan-Jittor | 保研面试 |

 ## 人像属性细粒度编辑

| Year | Venue   | Model      | Document Title                                           | 代码地址                              | 事由       |
| ---- | -----   | ---------- | -------------------------------------------------------- | ------------------------------------- | ---------- |
| 2020 | CVPR | HiSD | Fine-grained Portrait Attribute Editing | https://github.com/XLINYIN/HiSD-Jittor | 保研面试 |