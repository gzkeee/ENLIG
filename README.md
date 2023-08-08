# Enhancing Pre-trained Language Models with Knowledge Representation Using Line Graphs
In this paper, we introduces a novel knowledge representation approach that effectively captures the connection patterns of relations surrounding entities in knowledge graphs. By employing line graphs, we can explicitly model correlations among relations, enhancing the representation of entity knowledge and reduce the number of parameters.



## Environment

1. **Hardware Configuration:**
   - CPU: Intel Core i7-10700K (8 cores, 3.8 GHz)
   - GPU: NVIDIA GeForce RTX 3090
   - Memory: 32 GB DDR4 RAM
   - Storage: 1 TB NVMe SSD
   
2. **Software Configuration:**
- Operating System: Ubuntu 20.04 LTS
  
- Programming Language: Python 3.8
  
- Deep Learning Frameworks: PyTorch 1.9.0
  
- Huggingface Transformers 4.31.0

## Experience

### Prepare Pre-train Data

```bash
python3 process_data.py
```

### Pretrain

````bash
python3 pretrain.py
````

### Fine-tuning on Downstream Tasks

#### Relation Extraction

````bash
python3 run_fewrel.py
````

#### Entity Typing

```bash
python3 run_typing.py
```

#### Question Answering over KG

```bash
python3 run_metaQA.py
```

