# DLS-image2image-project
## Final image2image project of first half of Deep Learning School 
### Examples of inference: 

![](https://sun9-12.userapi.com/impf/oRqVmOB-Zwo3zbxNbToxFwEgchMFQsA85SPRVg/5UQNYxo7h3A.jpg?size=256x256&quality=96&sign=af7a6b6ab767acc83bef7b7bbd48f9c3&type=album)
![](https://sun9-4.userapi.com/impf/qUTt21T_3gi3NHtWPr_RBThrrnzDSG_RawxQhw/wQT8Gpf5RmI.jpg?size=256x256&quality=96&sign=f9cc2a599ae01e2588609a8de60f7918&type=album)

![](https://sun9-48.userapi.com/impf/Gck886uJ29yWMpvOyDOxFpGbCg9zWH8x_jLlig/36htaxZ-urY.jpg?size=256x256&quality=96&sign=695cffa61420ec9d21a7946e93c384e6&type=album)
![](https://sun9-41.userapi.com/impf/vpkmfsiqGNkEsOXDsmAl9iv_OF44O8K5GKQodA/CpjPqIIEaNY.jpg?size=256x256&quality=96&sign=aff009c0a29b25724bf488fecc732962&type=album)


![](https://sun9-44.userapi.com/impf/DMOhA5H0ZRRyxTiPucnd9FqKk5tQKdzEj6iYIA/f4PCScX70ak.jpg?size=256x256&quality=96&sign=0cc0b7772028e19d0245b18e0cc78da0&type=album)
![](https://sun9-50.userapi.com/impf/Z9Wd06l_zpHUauBOwJefWdpfViJlp4aDEuvXYw/6jFnOhAG1WY.jpg?size=256x256&quality=96&sign=9e33dbdf26a0e273ca7c422e79ce885d&type=album)
___
### Information for reviewer
Мои комментарии по работе в конце ноутбука Baseline_Monet
___
### How to use
- clone repo
- make venv and activate
- pip install -r requirements.txt
- open bot/app.py
- insert your token into `bot = Bot("INSERT_YOUR_TOKEN")`
- make sure that you have genB_weights.tar inside your work directory*
- run app.py**
- wait from 1 to 5 min to ensure your tg bot activated
- use `/start` command in your tg bot to get instuction or just send 1 pic of female fullface
- wait answer

Example of usage: 

![](https://sun9-79.userapi.com/impf/VuUMwpUxReaONIAzguJzMmmU58n4fLxX48kHMw/dD4AdHtoWqc.jpg?size=614x972&quality=96&sign=39134e3fb845da7b93d5a867c51e0a16&type=album)

\* - if you've got problem getting it from github you can take it and others weights [here](https://disk.yandex.ru/d/-NmOZAwlLGJCIQ)

\** - if you want to train get all weights from link above, put them in working directory, make 2 directories "trainA" and "trainB" with datasets, and empty directory "training_samples" and run train.py from animefication folder
