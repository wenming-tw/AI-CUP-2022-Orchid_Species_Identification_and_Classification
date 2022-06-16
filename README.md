## Competition
[尋找花中君子 - 蘭花種類辨識及分類競賽](https://tbrain.trendmicro.com.tw/Competitions/Details/20)  
隊伍名稱：TEAM_1123

## Leaderboard
```
Leaderboard |   Score   |   Rank
----------------------------------
Public      | 0.921221  |   10th
Private     | 0.811896  |   10th
```

## 套件
```
Package         |   Version
--------------------------------
torch           |   1.11.0+cu113
torchvision	    |   0.12.0+cu113
numpy	        |   1.21.2
pandas	        |   1.3.4
transformers	|   4.18.0
opencv-python	|   4.5.5.64
albumentations	|   1.1.0
tqdm	        |   4.62.3

```

## 模型訓練
執行後會產生 Log 檔、Checkpoint 檔、Tensorboard 檔。

```python
python main_iter.py
```

## 產生預測結果
執行後會產生 5 個Fold的結果檔 (xxx_x.csv)、機率值檔(.pt)，與集成的結果檔(xxx.csv)。
```python
python test.py
```

## 資料夾結構
請將資料放置到正確的資料夾內。
```
├── code
│   ├── main_iter.py
│   ├── test.py
│   └── utils.py	
├── data
│   ├── training
│   │   ├── label.csv
│   │   ├── zyebproc9x.jpg
│   │	└── ....		
│   ├── orchid_public_set
│   │  ├── 0a3wry7o4s.jpg
│   │  ├── 0a5ry496dc.jpg
│   │  └── ....					
│   ├── orchid_private_set
│   │  ├── 0a2xqs6vrl.jpg
│   │  ├── 0a3nyp8hie.jpg
│   │  └── ....			
│   └── submission_template.csv
├── logs
│   ├── 04_30_2022_22_50_18.log
│	└── ....
├── checkpoint
│   ├── 04_30_2022_22_50_18
│   │   ├── 0.ckpt
│   │   ├── 1.ckpt
│   │   ├── 2.ckpt
│   │   ├── 3.ckpt
│   │   └── 4.ckpt
│	└── ....
├── outputs
│   ├── 04_30_2022_22_50_18_0.csv
│	├── 04_30_2022_22_50_18_0.pt
│   ├── 04_30_2022_22_50_18_1.csv
│	├── 04_30_2022_22_50_18_1.pt
│   ├── 04_30_2022_22_50_18_2.csv
│	├── 04_30_2022_22_50_18_2.pt
│   ├── 04_30_2022_22_50_18_3.csv
│	├── 04_30_2022_22_50_18_3.pt
│   ├── 04_30_2022_22_50_18_4.csv
│	├── 04_30_2022_22_50_18_4.pt      
│   └── 04_30_2022_22_50_18.csv           
└── tensorboard
    ├── 04_30_2022_22_50_18
    │   ├── fold_0
    │   ├── fold_1
    │   ├── fold_2
    │   ├── fold_3
    │   └── fold_4
	└── .... 
```
