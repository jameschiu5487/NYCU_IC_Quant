import twstock
import pandas as pd

target_stock = '3105' 
stock = twstock.Stock(target_stock) 
target_price = stock.fetch_from(2016, 1)  #取用2020/05至今每天的交易資料
name_attribute = [
    'Date', 'Capacity', 'Turnover', 'Open', 'High', 'Low', 'Close', 'Change',
    'Transcation'
]  
df = pd.DataFrame(columns=name_attribute, data=target_price)
#將twstock抓到的清單轉成Data Frame格式的資料表
filename = f'{target_stock}.csv'
df.to_csv(filename)
#將Data Frame轉存為csv檔案
