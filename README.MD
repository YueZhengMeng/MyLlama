## 手搓Llama模型,个人学习用

## 1. 项目介绍
与其说是手搓,不如说是基于transformers==4.36.0的llama源码进行简化与重构  
保留了LlamaForCausalLM的核心结构与计算逻辑,并且进行了详细的注释  
手搓的模型在MyLlama.py文件中, 去掉了所有transformers的依赖,仅基于pytorch即可运行  
目前仅实现了与transformers版LlamaForCausalLM的参数互相加载,以及相同参数下,相同的输入可以推理得到相同的输出   
训练相关的代码,以后有时间再完善

## 2. 项目结构
MyLlama.py:              手搓的Llama模型文件  
run_official_model.py:   运行transformers版模型的,分别使用generate函数与手生成搓函数调用,并保存初始化后的模型参数  
run_my_model.py:         运行手搓的模型,加载transformers版模型的参数,使用手搓生成函数调用,验证推理结果是否一致  
RoPE.ipynb:              RoPE旋转位置编码的原理,以及Meta版和transformers版的两种实现方法的详解  
llama_struct.svg:        B站up主良睦路程序员制作的transformers版llama模型的结构示意图,用于辅助理解

## 3. 一些细节问题
transformers版llama模型的KV-Cache机制的代码,由于历史版本的兼容性问题,以及新版本pytorch算子的兼容性问题  
cache的传递逻辑存在不影响运行,但是绕了一个圈子的问题  
详细问题描述见MyLlama.py文件231行到237行的注释  
即使是目前最新版的源码中,KV-Cache的机制还处于TODO待重构状态  
我也就先保留原有的逻辑,以后有时间再进行更新

LlamaDecoderLayer的输出,也存在返回值的数量与形状不确定和不严格的问题,后面的代码写了一个判断来适配  
我也保留原有的逻辑,以后有时间再进行更新

## 4. 个人建议
想要学习llama源码的同学,可以先去看看[Meta官方版的源码](https://github.com/meta-llama/llama3/blob/main/llama/model.py)  
Meta官方版的源码只有300行,仅基于pytorch,结构清晰,逻辑简单,非常适合学习  
transformers版的源码,等自己熟悉了transformers框架,以及transformers官方重构完成之后再看
