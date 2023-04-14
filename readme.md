# frontend application

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
OS: Ubuntu 16.04.4 LTS
Gcc version: 5.4.0 20160609 (Ubuntu 5.4.0-6ubuntu1~16.04.10)
GPU: TITAN Xp
CUDA: 8.0

### reference
If you found this repo useful, please consider citing our paper:
