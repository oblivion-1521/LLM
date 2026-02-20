目前已经使用的加速：
1. FP16半精浮点
2. TensorFloat-32的Ampere架构牺牲精度加速
3. torch.inference_mode()关闭梯度计算
4. 使用KV Cache(use_cache=True)

其他加速空间：
- 主要是python**调度开销**和**显存读写**
1. Flash Attention 2：对 IO 优化的注意力算法，需要pip install flash-attn --no-build-isolation
2. torch.compile：PyTorch 2.0的编译器，将python**编译**成优化的CUDA Kernel，减少python interpreter's overhead.  
具体做法是在model.eval()后面加上model = torch.compile(model)

Accelerate Model weight loading:
- 使用.safetensors格式支持**内存映射（mmap）**，可以直接从磁盘映射到内存，速度最快。
- 丢掉device_map
- 架构改进，不重复加载，写成交互式脚本或API服务。



