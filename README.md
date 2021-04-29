# Fine Tuning Bert For Question Answering Using Tensorflow-2.x

Tech Stacks Use:

## 1. Tensorflow2.4
## 2. Tensorflow-Hub
## 3. Hugging Face Tokenizers
## 4. Fast-Api
## 5. Google Colab Pro (for training)
## 6. Azure VMs (for deployment)

For fine tuning I've used Bert Model from Tensorflow Hub. For tokenization I've used BertWordPieceTokenizer from Tokenizers library in HuggingFace.
Dataset Used: Squad-v1 dataset (Can be found in the ./Datasets folder)

For training I've used Google Colab Pro, and got the opportunity to test out Tesla V100-SXM2 GPU.

# Here's a snap of the GPU consumption while training.

![](https://github.com/Swapnadeep1998/Question_Answering_BERT/blob/main/Images_Screeshots/Screenshot%20from%202021-04-28%2002-09-28.png)



The trained model was aroung 1.4gb and hence didn't upload it on GitHub. 

I've used FastApi to build the API for the model and successfully deployed it on Azure VMs. 

Here are few snaps of how good is the model in extracting answers for a question out of a given context (Paragraph).
I took the advantage of the built in Swagger UI in FastAPI.

# Example 1

![](https://github.com/Swapnadeep1998/Question_Answering_BERT/blob/main/Images_Screeshots/Screenshot%20from%202021-04-29%2018-01-32.png)


![](https://github.com/Swapnadeep1998/Question_Answering_BERT/blob/main/Images_Screeshots/Screenshot%20from%202021-04-29%2018-02-04.png)



# Example 2


![](https://github.com/Swapnadeep1998/Question_Answering_BERT/blob/main/Images_Screeshots/Screenshot%20from%202021-04-29%2020-55-28.png)


![](https://github.com/Swapnadeep1998/Question_Answering_BERT/blob/main/Images_Screeshots/Screenshot%20from%202021-04-29%2020-55-47.png)
