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
parameters of decoder, encoder and attention decoder are initialized using a uniform distribution with a range of -0.8 to 0.8 using the function init.uniform_(param, -0.8, 0.8).

#### Hyperparameters for dec_lstm:
- word_embedding_size: The size of the word embedding used in the model. Default value is 300.
- rnn_size: The size of the recurrent hidden state in the LSTM. This affects the capacity of the model and may impact its performance. You can specify the value of rnn_size when initializing the model. Default is 300.
- dropout_de_out: The dropout rate applied to the output of the LSTM layer. A higher value of dropout_de_out can increase regularization and reduce overfitting, but may also result in underfitting. Default value is 0, meaning no dropout is applied. You can specify a custom value for dropout_de_out when initializing the model.

#### Hyperparameters for decoder:
- rnn_size: The size of the recurrent hidden state in the LSTM. This affects the capacity of the model and may impact its performance. You can specify the value of rnn_size when initializing the model. Default is 300
- word_embedding_size: The size of the word embedding used in the model. Default value is 300.
- dropout_de_in: The dropout rate applied to the input of the LSTM layer. A higher value of dropout_de_in can increase regularization and reduce overfitting, but may also result in underfitting. Default value is 0, meaning no dropout is applied. You can specify a custom value for dropout_de_in when initializing the model.
- input_size: The size of the input vocabulary, which determines the number of unique words in the input data. You need to specify the value of input_size when initializing the model. Default will be vocabulary size generated after preprocessing.

#### Hyperparameters:
- rnn_size: The size of the recurrent hidden state in the LSTM. This affects the capacity of the model and may impact its performance. You can specify the value of rnn_size when initializing the model. Default is 300
- separate_attention: A boolean flag indicating whether to use separate attention or not. Default value is True. When set to True, a separate linear layer is used for attention computation. You can modify this flag to True or False depending on your requirements.
- output_size: The size of the output vocabulary, which determines the number of unique words in the output data. You need to specify the value of output_size when initializing the model. Default will be vocabulary size generated after preprocessing.
- dropout_for_predict: The dropout rate applied to the output of the AttnUnit layer during prediction. A higher value of dropout_for_predict can increase regularization and reduce overfitting, but may also result in underfitting. Default value is 0, meaning no dropout is applied. You can specify a custom value for dropout_for_predict when initializing the model.


### reference
If you found this repo useful, please consider citing our paper:
