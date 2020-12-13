# Profs of improvement
Do vấn đề overfit, để chứng minh cải tiến so với model gốc, ta xem xét khả năng của model trong quá trình training, tức tốc độ hội tụ và F1 score trên training set.  
Màu xanh lam : Model với multihead attn question-context + multihead attn question-question trên bilstm  
Màu xanh lá  : Model với multihead attn question-context
Màu xanh hồng: Original model  
![alt text](https://github.com/NNHieu/TrueAnswer_ZaloAIChallenge2019_VietnameseWikiQA/tree/master/images/TrainLoss.png "Train loss")  
![alt text](https://github.com/NNHieu/TrueAnswer_ZaloAIChallenge2019_VietnameseWikiQA/tree/master/images/F1.png "Train loss")  
![alt text](https://github.com/NNHieu/TrueAnswer_ZaloAIChallenge2019_VietnameseWikiQA/tree/master/images/ValLoss.png "Train loss")  

# Setup
Yêu cầu:
* docker

Setup env:
* Build docker image: `docker-compose build gpu`
* Create docker container: `docker-compose build gpu` và truy cập jupyter notebook server với address có trong terminal

Training:
* Run notebook `DrQA.ipynb` với các config (dataset đã được chuẩn bị trước, xem kĩ hơn tại `dataset/qaset/`)

Logger:
* Run notebook `TensorboardLogger`, cần đặt tên cho experiemnt khi training để dễ phân biệt