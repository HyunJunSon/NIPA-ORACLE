import requests
from bs4 import BeautifulSoup
from openpyxl import Workbook

url = 'https://finance.naver.com/'

response = requests.get(url)
response.raise_for_status()
html = response.text
soup = BeautifulSoup(html, 'html.parser')

# 인기검색종목 (10개)
tbody1 = soup.select_one('#container > div.aside > div.group_aside > div.aside_area.aside_popular > table > tbody')
trs1 = tbody1.select('tr')

# 시가총액 상위종목 (10개)  
tbody2 = soup.select_one('#container > div.aside > div.group_aside > div.aside_area.aside_market_sum > table > tbody')
trs2 = tbody2.select('tr')

datas = []

# 인기검색종목 데이터
for tr in trs1:
    name = tr.select_one("th > a").get_text(strip=True)
    current_price = tr.select_one("td").get_text().replace(",","")
    change_direction = tr['class'][0]
    change_price = tr.select_one('td > span').get_text().strip()
    datas.append([name, current_price, change_direction, change_price])

# 시가총액 상위종목 데이터
for tr in trs2:
    name = tr.select_one("th > a").get_text(strip=True)
    current_price = tr.select_one("td").get_text().replace(",","")
    change_direction = tr['class'][0]
    change_price = tr.select_one('td > span').get_text().strip()
    datas.append([name, current_price, change_direction, change_price])

writer_wb = Workbook()
writer_ws = writer_wb.create_sheet('결과')
writer_ws.append(['stock_name','stock_price','up_down','up_down_price']) 
for data in datas:
    writer_ws.append(data)
    
writer_wb.save(r'data/naver_fin_more.xlsx')
print(f"총 {len(datas)}개 데이터 수집 완료")
