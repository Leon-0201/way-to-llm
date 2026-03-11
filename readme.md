# 第一章 基础与原理 
## 1.1 深度学习基础
- 神经网络基础（前向传播、反向传播）
- 激活函数（Sigmoid、ReLU、GELU、SwiGLU）
- 损失函数（交叉熵、MSE、KL 散度）
- 优化器（SGD、Adam、AdamW、学习率调度）
- 正则化（Dropout、L2 正则、早停）
- 归一化（BatchNorm、LayerNorm、RMSNorm）
- 梯度问题（消失、爆炸、裁剪）
## 1.2 训练技术基础
- 混合精度训练（FP16、BF16、AMP）
- 梯度累积
- 梯度检查点（Gradient Checkpointing）
- 检查点机制（Checkpoint、训练中断与恢复）
- 数据加载与预处理

---
# 第二章 模型架构 
## 2.1 Transformer 核心
- Self-Attention 机制
- 多头注意力（Multi-Head Attention）
- 前馈网络（FFN、SwiGLU）
- 残差连接（Residual Connection）
- 归一化位置（Pre-Norm vs Post-Norm）
- 编码器 - 解码器架构
## 2.2 位置编码
- 绝对位置编码（Learned、Sinusoidal）
- 相对位置编码（RoPE、ALiBi）
- RoPE 原理与实现（LLaMA 采用）
- 上下文扩展（NTK-Aware、YaRN）
## 2.3 分词器（Tokenizer）
- BPE（Byte Pair Encoding）
- WordPiece、Unigram
- 特殊 token（BOS、EOS、PAD、UNK）
- 词表大小与影响
## 2.4 注意力优化
- KV Cache 原理与优化
- Flash Attention（v1/v2）
- PagedAttention（vLLM 核心）
- 线性注意力（Linear Attention）
- 稀疏注意力（Sparse Attention）
## 2.5 大模型架构演进
- Decoder-only 架构（GPT 系列）
- LLaMA 架构详解
- MoE 架构（Mixtral、GPT-4 MoE）

---
# 第三章 训练与微调 
## 3.1 预训练（Pretraining）
- 数据收集与清洗
- 数据配比与质量过滤
- 训练目标（Next Token Prediction）
- 训练曲线与收敛判断
## 3.2 分布式训练并行技术
- 数据并行（DDP、FSDP）
- 张量并行（Tensor Parallel）
- 流水线并行（Pipeline Parallel）
- 序列并行（Sequence Parallel）
- ZeRO 优化（ZeRO-1/2/3）
## 3.3 训练框架
- DeepSpeed
- Megatron-LM
- PyTorch FSDP
- ColossalAI
## 3.4 全参数微调（Full Fine-tuning）
- 适用场景与资源需求
- 数据准备与格式
## 3.5 参数高效微调（PEFT）
- LoRA 原理与实现
- QLoRA（量化 + LoRA）
- Prefix Tuning
- Prompt Tuning
- P-tuning / P-tuning v2
- Adapter 方法
- 多任务学习
## 3.6 对齐与偏好优化
- SFT（Supervised Fine-Tuning）
- 奖励模型训练（Reward Model）
- RLHF（PPO、Reward Modeling）
- 偏好优化（DPO、ORPO、IPO）
## 3.7 知识蒸馏
- 蒸馏原理（Teacher-Student）
- 损失函数（KL 散度、交叉熵）
- 应用场景（模型压缩）
## 3.8 微调框架
- LLaMA Factory
- HuggingFace PEFT
- Axolotl

---
# 第四章 部署与推理 
## 4.1 推理框架
- vLLM（PagedAttention、continuous batching）
- Ollama（本地部署）
- Text Generation Inference（TGI）
- TensorRT-LLM
- llama.cpp（CPU 推理）
## 4.2 推理性能指标
- 首 Token 延迟（TTFT）
- Token 生成速度（Tokens/s）
- 吞吐量（Requests/s）
- 显存占用估算
- 计算量估算（FLOPs）
## 4.3 显存与并行
- 显存组成（权重、KV Cache、激活值）
- 模型并行策略选择
- 通信开销（NVLink、InfiniBand）
## 4.4 量化技术 
- 量化原理（INT8、INT4、FP8）
- PTQ（Post-Training Quantization）
- QAT（Quantization Aware Training）
- GPTQ、AWQ、SmoothQuant
- 量化对精度的影响
## 4.5 推理优化技术
- 投机采样（Speculative Decoding）
- 批处理优化（Continuous Batching）
- 模型融合与算子优化
- 显存优化（Offloading、Swap）
## 4.6 服务化
- API 设计（REST、gRPC）
- 负载均衡
- 多租户与配额管理
- 请求队列与限流

---
# 第五章 提示词工程 
## 5.1 基础技巧
- Zero-shot / Few-shot
- 给定身份/角色
- 任务拆分与逐步推理
- 给模型思考时间
## 5.2 推理增强
- CoT（Chain of Thought）
- ToT（Tree of Thought）
- GoT（Graph of Thought）
- Self-Consistency
## 5.3 结构化输出
- JSON Schema 约束
- 函数调用格式
- XML/Markdown 格式控制
## 5.4 多轮对话
- 对话历史管理
- 系统指令（System Prompt）
- 上下文窗口管理
## 5.5 安全与越狱
- 常见越狱手法
- 防御策略
- 内容安全过滤
## 5.6 提示词优化
- 自动优化（AutoPrompt）
- 评估与迭代

---
# 第六章 Agent 与 RAG 
## 6.1 Agent 基础
- Agent 定义与组成
- 设计范式（ReAct、Plan-and-Solve）
- 常见框架（LangChain、LangGraph、AutoGen）
## 6.2 Agent 核心能力
- 任务规划与分解
- 工具使用与选择
- 自我反思与修正
- 多 Agent 协作
## 6.3 工具集成
- Function Call / Tool Use
- MCP（Model Context Protocol）
- Skills/Plugins
- API 集成
## 6.4 上下文管理
- 长短期记忆机制
- 上下文压缩与摘要
- Lost in the Middle 问题与解决
## 6.5 RAG 基础 
- RAG 工作流程
- 应用场景与优势
## 6.6 RAG 数据处理
- 数据收集与清洗
- 文档切片策略（固定长度、语义切片）
- 元数据管理
## 6.7 向量化
- 嵌入模型选型（text-embedding、bge、m3e）
- 多模态向量化
- 向量维度与相似度计算
## 6.8 向量数据库
- 选型对比（Milvus、Pinecone、Weaviate、Chroma）
- 索引类型（HNSW、IVF）
- 混合检索（向量 + 关键词）
## 6.9 检索优化
- 检索策略（Top-K、阈值）
- 重排序（Rerank）
- 多路检索与融合

---
# 第七章 落地与实践 
## 7.1 落地流程
- 需求分析与场景评估
- 意图识别与路由
- 模型选型（开源 vs 闭源）
- POC 验证
## 7.2 输出控制
- 温度（Temperature）
- Top-K / Top-P 采样
- 重复惩罚
- 结构化输出保证
## 7.3 幻觉问题
- 幻觉类型与原因
- 检测与评估
- 缓解策略（RAG、引用来源、验证）
## 7.4 评估体系
- 自动评估（BLEU、ROUGE、BERTScore）
- 大模型评估（LLM-as-a-Judge）
- 人工评估标准
- A/B 测试设计
## 7.5 监控与运维
- 请求追踪与日志
- 性能监控（延迟、吞吐量）
- 成本分析与优化
- 错误处理与降级
## 7.6 安全与合规
- 数据隐私
- 内容审核
- 模型滥用防护