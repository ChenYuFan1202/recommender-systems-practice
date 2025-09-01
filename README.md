> 本專案基於 [recommender-tutorial](https://github.com/topspinj/recommender-tutorial) 進行修改。  
> 原始版本由 Jill Cates 開發，依據 BSD 3-Clause License 授權使用。

# 虛擬環境建立（使用 conda）

### 建立環境（使用 Python 3.12）
```bash
conda create -n recommender_systems python=3.12 # n 為 name 的意思 
                                                # 也可以輸入 conda create --name recommender_systems python=3.12
conda activate recommender_systems
pip install -r requirements.txt # -r 是 requirements(s) 的縮寫 # 有問題再手動調整
```

# Recommendation Systems 101

This series of tutorials explores different types of recommendation systems and their implementations. Topics include:

- collaborative vs. content-based filtering
- implicit vs. explicit feedback
- handling the cold start problem
- recommendation model evaluation

We will build various recommendation systems using data from the [MovieLens](https://movielens.org/) database. You will need Jupyter Lab to run the notebooks for each part of this series. Alternatively, you can also use Google’s new [colab platform](https://colab.research.google.com) which allows you to run a Jupyter notebook environment in the cloud. You won't need to install any local dependencies; however, you will need a gmail account. 

The series is divided into 3 parts:

1. [Building an Item-Item Recommender with Collaborative Filtering](#part-1-building-an-item-item-recommender-with-collaborative-filtering)
2. [Handling the Cold Start Problem with Content-based Filtering](#part-2-handling-the-cold-start-problem-with-content-based-filtering)
3. [Building an Implicit Feedback Recommender System](#part-3-building-an-implicit-feedback-recommender-system)


More information on each part can be found in the descriptions below.

### Part 1: Building an Item-Item Recommender with Collaborative Filtering

| |Description |
|:-----------|:----------|
|Objective|Want to know how Spotify, Amazon, and Netflix generate "similar item" recommendations for users? In this tutorial, we will build an item-item recommendation system by computing similarity using nearest neighbor techniques.|
|Key concepts|collaborative filtering, content-based filtering, k-Nearest neighbors, cosine similarity|
|Requirements|Python 3.6+, Jupyter Lab, numpy, pandas, matplotlib, seaborn, scikit-learn|
|Tutorial link|[Jupyter Notebook](part-1-item-item-recommender.ipynb)|
|Resources|[Item-item collaborative filtering](https://www.wikiwand.com/en/Item-item_collaborative_filtering), [Amazon.com Recommendations](https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf), [Various Implementations of Collaborative Filtering](https://towardsdatascience.com/various-implementations-of-collaborative-filtering-100385c6dfe0) |

### Part 1：Item-Item-Recommender 所學到的內容
- Bayesian average：處理評分很高或很低但數量很少的狀況。  
- Collaborative Filtering (CF)：是非監督式學習。  
- User-item matrix (utility matrix)：雖然有 user-item 這個詞，但這是一個矩陣而不是演算法。  
- Manhattan distance：abs(x1 - x2) + abs(y1 - y2)。  
- 更深的理解 user-based 和 item-based，前者為推薦相似的人喜歡的內容、後者為推薦類似的物品，不過兩種都會用使用者的 Feedback。  
- 最後 Extra：User-based 的內容是我額外想做的，因為原作者接下來是直接進到 content-based 的內容。  

**補充：**   
- 由於我是先 Clone 作者的 repo，而不是 Fork，通常沒有推送權限，需要手動將 origin 改成自己的 repo 才能 push。因此，會需要輸入 `git remote set-url origin <url>`，將 origin 改為自己的 GitHub repo 位置。  
- Fork 和 Clone 的不同之處在於 Fork 會複製一份作者的 repo 到自己的帳號並將遠端（origin）指向自己的 repo，因此對自己的 Fork 擁有推送權限。要注意的是，還是需要用 `git clone` 下載 Fork。

### Part 2: Handling the Cold Start Problem with Content-based Filtering

| |Description |
|:-----------|:----------|
|Objective|Collaborative filtering fails to incorporate new users who haven't rated yet and new items that don't have any ratings or reviews. This is called the cold start problem. In this tutorial, we will learn about clustering techniques that are used to tackle the cold start problem of collaborative filtering.|
|Requirements|Python 3.6+, Jupyter Lab, numpy, pandas, matplotlib, seaborn, scikit-learn|
|Tutorial link|[Jupyter Notebook](part-2-cold-start-problem.ipynb)|

### Part 2：Content-based Filtering 所學到的內容  
- Content-Based Filtering 一般被視為非監督式學習，因為它主要依靠物品或使用者特徵向量與相似度計算。但在實務中，也能結合監督式方法（例如分類或排序模型）來提升效果。
- 我發現原作者三個做錯的地方並進行修正  
    1. (no genres listed) 沒有正確刪除。  
    2. 沒有發行年份的電影只有四個，作者的函式邏輯是對的，但 Series 會在後面新增 `""`。  
    3. 作者使用 `movies.index` 將 index 轉換成原始的電影名稱，但因為前兩步驟有刪除電影，因此，會發生問題。  
- Extra：我修改原作者最後的函式，將推薦的電影附上風格以及相似度分數，雖然相似度分數在實際畫面不該出現，但這邊是為了要做驗證。  
- 結論：學到很多 Content-Based Filtering 特徵處理的步驟，如果有使用者特徵也可以這樣做推薦，很有趣，謝謝原作者！

### Part 3: Building an Implicit Feedback Recommender System

| |Description |
|:-----------|:----------|
|Objective|Unlike explicit feedback (e.g., user ratings), implicit feedback infers a user's degree of preference toward an item by looking at their indirect interactions with that item. In this tutorial, we will investigate a recommender model that specifically handles implicit feedback datasets.|
|Requirements|Python 3.6+, Jupyter Lab, numpy, pandas, implicit|
|Tutorial link|[Jupyter Notebook](part-3-implicit-feedback-recommender.ipynb)|

### Part 3：Implicit Feedback Recommender System 所學到的內容  
- 做法和 CF 很像，但用不同的特徵和模型。  
- Matrix factorization is particularly useful for very sparse data（這一點讓我知道線性代數的重要，需要再補齊!）。  
- 有 ALS（Alternating Least Squares）和 BPR（Bayesian Personalized Rankin） 等作法。  
- similar_items() 和 recommend()，前者依照物品推、後者依照推使用者推。  
- 最大的雷點是要記得模型用什麼當列訓練就要用什麼列來預測。  
- Extra：我有補上風格來驗證原作者的說法，但似乎不太正確。  

**補充：**  
- 我在無法使用的圖片或連結都有說明無效。