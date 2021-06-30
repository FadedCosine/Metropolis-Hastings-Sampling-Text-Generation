# 基于Metropolis-Hastings采样的带约束语句生成方法

## 文件结果
该项目上传的文件结构如下：

```
├── data # 包含进行实验的数据
├── model # 包含预训练的模型文件
├── bertmlm.py # 对BERT进行fine-tune的代码
├── data_process.py # 包含关键词选取的代码
├── eval_ppl_by_gpt2.py # 使用GPT-2作为第三方语言模型进行PPL测试的代码
├── gptlm.py # 对GPT-2进行fine-tune的代码
├── mhsg.py # 使用Metropolis-Hastings采样方法进行文本生成的框架代码
├── README.md
├── run_1_billion.py # 在One Billion语料库中进行实验的主程序
├── run_CommonGen.py # 在CommonGen数据集中进行实验的主程序
├── run.sh # 程序运行的sh文件
└── utils.py 
```

## 运行方法
直接运行run.sh文件
```
sh run.sh
```