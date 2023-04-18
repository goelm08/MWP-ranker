# frontend application

The app is deployed at https://tekstar.in/mwpranker.php
### start the frontend application

```bash
cd frontend
npm i
npm start
```

# backend application

1. preprocess.ipynb - preprocess the equation data into tree format 

### training the Graph2Tree model

#### start the standford coreNLP server
1. ```bash echo "standford-core-nlp"```
2. ```bash cd stanford-corenlp-4.5.1``` 
3. ```bash java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 1500000```

1. Train 
```bash cd backend && sh train.sh```

2. Test
``` bash cd backend && sh test.sh```

3. Test with your own trained model
```bash cd backend && pip install -r requirements.txt && && cd src python pipeline.py```

### environment

1. OS: Ubuntu 20.04.2 LTS
2. G++ version: 9.4.0
3. CPU: AMD EPYC 7763 64-Core Processor
3. GPU: NVIDIA A100-SXM4-40GB
4. CUDA: 11.6

### hyperparameters for model
It is initialized using a uniform distribution with a range of -0.8 to 0.8 using the function init.uniform_(param, -0.8, 0.8).


### reference
If you found this repo useful, please consider citing our paper:
