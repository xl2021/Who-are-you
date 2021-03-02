# Who-are-you

## 1.Background
As the returner of Chang'e-5 successfully landed on Siziwang Banner, the small animal appeared in front of it caught people's eyes. They were interested in what the animal was. In order to figure out the answer, this project tries to identify the animal using machine learning.

## 2.Dataset
Since there is no dataset available for this task, the project selects 11 possible species which are mentioned for the most times according to the guesses under [the question at Zhihu](https://www.zhihu.com/question/435202802). Next, a crawler is applied to search and download pictures describing these species from network. These pictures are manually cleaned to improve their quality and maintain the balance across different species. The species and their numbers are shown as following:

 animal | # of train set | # of validation set |　# of test set | # of all 
 ------ | -------------- | ------------------- | -------------- | --------
 cat | 288 | 95 | 95 | 478 
 desert_fox | 312 | 102 | 100 | 514 
 dog | 318 | 106 | 106 | 530 
 manul | 324 | 108 | 108 | 540 
 marmot | 308 | 102 | 102 | 512 
 marten | 312 | 104 | 100 | 516 
 mongolia_rabbit | 324 | 96 | 96 | 516 
 rabbit | 306 | 100 | 100 | 506 
 red_fox | 275 | 90 | 90 | 455 
 weasel | 301 | 100 | 100 | 501 
 wolf | 300 | 98 | 96 | 494 

## 3.Model
A LeNet-5-like neural network is used. A major difference is that the input shape is modified to 68 * 68 * 1 because it is similar to the original size of animal appearing in the published video and the difference among the values of R, G and B channels is pretty small so that the frames are more similar to grayscale images rather than colorful pictures. As a result, the pictures in dataset are converted into grayscale images while preprocessing. The details of neural network are as following:  
![Model](https://github.com/xl2021/Who-are-you/blob/main/model/model.png)

## 4.Training
The model is trained on a Celeron N2940 CPU for 50 epoches, the final accuracy on train, validation and test set is 0.9676, 0.4023 and 0.3998, respectively, where dramatical overfitting appears here. 

## 5.Result
For the purpose of accuracy, an algorithm is designed to determine the exact position of the animal before cropping and fetching it into the model. The final prediction is:

 Top N | prediction* 
 ------| ------------ 
 Top 1 | cat(124/306) 
 Top 2 | cat(211/306) 
 Top 3 | cat(269/306) 
 Top 4 | cat(290/306) 
 Top 5 | cat(301/306) 
 Top 6 | cat(304/306) 
 Top 7 | cat(306/306) 
 
 *The prediction is the species that appears for the most times in the set consisting of top-N species in all of known frames.

Now consider the accuracy of this prediction based on test set. The table below shows given images belonging to the same species, what will the model predicts for top-N: 

 Species | Top 1 | Top 2 | Top 3 | Top 4 | Top 5 | Top 6 | Top 7 | Top 8 | Top 9 | Top 10 | Top 11 
 ------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ------ | ------ 
 cat | cat | cat | cat | cat | cat | cat | cat | cat | cat | cat * | - 
 desert_fox | desert_fox | desert_fox | desert_fox | desert_fox | desert_fox | red_fox | red_fox | red_fox | red_fox | red_fox | rabbit * 
 dog | dog | dog | dog | dog | dog | dog | dog | dog | weasel * | - | - 
 manul | manul | manul | manul | manul | manul | manul | cat | cat | cat | cat * | - 
 marmot | marmot | marmot | marmot | marmot | marmot | marmot | marmot | marmot | red_fox | manul * | - 
 marten | marten | marten | marten | marten | marten | marten | marten | cat | weasel | red_fox | desert_fox * 
 mongolia_rabbit | mongolia_rabbit | mongolia_rabbit | mongolia_rabbit | mongolia_rabbit | mongolia_rabbit | wolf | wolf | wolf | wolf | wolf | mongolia_rabbit * 
 rabbit | rabbit | rabbit | rabbit | rabbit | rabbit | rabbit | rabbit | rabbit | rabbit | red_fox * | - 
 red_fox | red_fox | red_fox | red_fox | red_fox | weasel | cat | cat | red_fox | cat | cat * | - 
 weasel | weasel | weasel | weasel | weasel | weasel | weasel | red_fox | red_fox | red_fox | weasel * | - 
 wolf | wolf | wolf | wolf | wolf | wolf | wolf | red_fox | wolf | manul | wolf | wolf * 
 
 *denotes that the species has appeared in all frames for top-N. 

From the table above we can find that even though the accuracy for a single image is less than 0.50 (approximately 0.40), the total reliability is very high given plenty of images belonging to the same species. 

In conclusion, the species that this model predicts is cat.

You can find the prediction for every frame [here](https://github.com/xl2021/Who-are-you/blob/main/demo/demo.mp4).

## 6.Further talk
Obviously, this project still needs to improve. The following are some possible directions.
 1. Overfitting
     As what we have noticed, dramatical overfitting appears while training. To overcome the problem, some strategies may be useful:
     * Collecting more pictures and expand the dataset; 
     * Trying some tricks for reducing overfitting; 
     * Reducing the number of parameters in model. 
 2. Dataset
     Since the frames in the published video are thermodynamic images, the dataset should consist of thermodynamic pictures as well. However, thermodynamic pictures are too rare to be found, so this project just finds some optical pictures and converts them to grayscale images to simulate thermodynamic images. As a result, the loss of accuracy is unavoidable. If there were a lot of thermodynamic images available, the prediction should be more accurate. 
 3. Model
     The architecture of model is pretty small and simple. I have tried some more big or complex model, but it does not work in terms of noticeably improving accuracy while the speed of training is much slower on my computer. But it is worthwhile to try other architectures, especially when both the quality and the quantity of dataset are improved.
