# 專案名稱
24/72-hour forecasting for PM2.5 by iTransformer with entmax

## 專案簡介

本專案使用 iTransformer 預測全臺灣 77 座空氣品質監測站的未來 24/72 小時 PM2.5 濃度。

## 最終成績

| 預測時長 | Avg MSE  | Avg MAE  | Avg R²   |
|-----------|----------|----------|----------|
| 24 小時   | 42.4625  | 4.6362   | 0.4676   |
| 72 小時   | 56.0082  | 5.4249   | 0.3063   |

---

## 測試資料說明

- 測試資料期間：**2025 年 1 月 ~ 11 月**
- 資料來源：**[環境部環境資料開放平臺/空氣品質監測小時值資料](https://data.moenv.gov.tw/dataset/detail/AQX_P_13<img width="998" height="94" alt="image" src="https://github.com/user-attachments/assets/383f90f4-90e3-4ed8-a1df-d1a6260aa9b9" />
)**
- 測站數量：
  - 共 **77 個環保署中央固定式空氣品質監測站**  
  - 目前運作測站共 78 個
- 特殊測站說明：
  - **員林站**：排除，因缺失 2025 年以前的觀測資料
  - **萬里站**：包含在 `moenv_station.csv`，但自 **2025 年 5 月 1 日起停止空氣品質監測**



