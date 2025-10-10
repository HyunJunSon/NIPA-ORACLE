from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

def scrape_naver_finance_news():
    # Chrome 옵션 설정
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = None
    try:
        # Chrome 드라이버 실행
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

        # 웹페이지 로드
        driver.get("https://m.stock.naver.com/investment/news/flashnews")
        time.sleep(3)

        # BeautifulSoup으로 파싱
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # 뉴스 아이템 찾기
        news_items = soup.select('a.NewsList_link__EB1t5')
        
        print(f"✅ 총 {len(news_items)}개 뉴스 발견")
        
        for i, item in enumerate(news_items[:5]):  # 상위 5개만
            title_elem = item.select_one('strong.NewsList_title__v55VO')
            title = title_elem.get_text(strip=True) if title_elem else "제목 없음"
            
            link = item.get('href', '')
            if link and not link.startswith('http'):
                link = 'https://n.news.naver.com' + link
            
            print(f"\n[{i+1}] {title}")
            print(f"링크: {link}")
            print("-" * 50)

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    scrape_naver_finance_news()
