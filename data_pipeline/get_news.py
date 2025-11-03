import requests
from lxml import html
import pandas as pd
from datetime import datetime
from scipy import stats
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from googletrans import Translator
import sqlite3

try:

    tokenizer = BertTokenizer.from_pretrained("bert_model/saved_tokenizer")

    sentiment_model = BertForSequenceClassification.from_pretrained("bert_model/saved_model")

except Exception:
    tokenizer = BertTokenizer.from_pretrained("../bert_model/saved_tokenizer")

    sentiment_model = BertForSequenceClassification.from_pretrained("../bert_model/saved_model")

TARGET_CLASSES = [
        "grid-slot",
        "article",
        "article-list-item",
        "category-list-article",
        "d-flex",
        "align-items-center",
        "flex-md-row",
        "flex-column",
        "img-35"
    ]

TARGET_CLASSES_v2 = [
        "grid-slot",
        "article",
        "article-list-item",
        "category-list-article",
        "d-flex",
        "align-items-center",
        "flex-md-row",
        "flex-column",
        "img-0"
    ]

SENTIMENT_LABEL_MAPPING = {0 : "positive", 1 : "negative", 2: "neutral"}

def get_page_count(url):
    response = requests.get(url)
    response.raise_for_status()  

    tree = html.fromstring(response.content)

    tree = html.fromstring(response.content)

    elements = tree.xpath("//*[contains(concat(' ', normalize-space(@class), ' '), ' page-item ') and contains(concat(' ', normalize-space(@class), ' '), ' hidden-xs ') and @class='page-item hidden-xs' or @class='hidden-xs page-item']")

    return int(elements[0].text_content().strip())

def parse_dates(input_string):

    # Dictionary to map Hungarian month names to numbers
    hungarian_months = {
        "január": 1, "február": 2, "március": 3, "április": 4, "május": 5,
        "június": 6, "július": 7, "augusztus": 8, "szeptember": 9, "október": 10,
        "november": 11, "december": 12
    }

    date_time_part = input_string.split("|")[0].strip()

    parts = date_time_part.replace(".", "").split()
    year = int(parts[0])
    month_name = parts[1]
    day = int(parts[2])
    time = parts[3]

    month = hungarian_months[month_name.lower()]

    date_time_str = f"{year}-{month:02d}-{day:02d} {time}"

    date_time = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M")

    return date_time

def fetch_elements_by_classes(url, classes, classes_v2):
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            tree = html.fromstring(response.content.decode('utf-8'))
            
            class_conditions = " and ".join(f"contains(@class, '{cls}')" for cls in classes)
            xpath_expression = f"//article[{class_conditions}]"
            
            articles = tree.xpath(xpath_expression)
            
            if articles:
                results = []
                for i, article in enumerate(articles, 1):
                    article_id = article.get('data-article-id', 'N/A')
                    
                    img_url = article.xpath('.//img/@src')
                    img_url = img_url[0] if img_url else 'N/A'
                    
                    properties = article.xpath('.//p[@class="properties"]/text() | .//p[@class="properties"]/a/text()')
                    properties = ' '.join([prop.strip() for prop in properties if prop.strip()])
                    
                    title_elem = article.xpath('.//h3[@class="title"]/a')
                    title = title_elem[0].text_content().strip() if title_elem else 'N/A'
                    title_url = title_elem[0].get('href', 'N/A') if title_elem else 'N/A'
                    
                    description = article.xpath('.//p[@class="lines-3"]/text()')
                    description = description[0].strip() if description else 'N/A'

                    properties = parse_dates(properties)

                    results.append({
                        "article_number": i,
                        "id": article_id,
                        "image_url": img_url,
                        "properties": properties,
                        "title": title,
                        "title_url": title_url,
                        "description": description
                    })
                return results
            else:
                class_conditions = " and ".join(f"contains(@class, '{cls}')" for cls in classes_v2)
                xpath_expression = f"//article[{class_conditions}]"
                
                articles = tree.xpath(xpath_expression)
                
                if articles:
                    results = []
                    for i, article in enumerate(articles, 1):
                        article_id = article.get('data-article-id', 'N/A')
                        
                        img_url = article.xpath('.//img/@src')
                        img_url = img_url[0] if img_url else 'N/A'
                        
                        properties = article.xpath('.//p[@class="properties"]/text() | .//p[@class="properties"]/a/text()')
                        properties = ' '.join([prop.strip() for prop in properties if prop.strip()])
                        
                        title_elem = article.xpath('.//h3[@class="title"]/a')
                        title = title_elem[0].text_content().strip() if title_elem else 'N/A'
                        title_url = title_elem[0].get('href', 'N/A') if title_elem else 'N/A'
                        
                        description = article.xpath('.//p[@class="lines-3"]/text()')
                        description = description[0].strip() if description else 'N/A'

                        properties = parse_dates(properties)

                        results.append({
                            "article_number": i,
                            "id": article_id,
                            "image_url": img_url,
                            "properties": properties,
                            "title": title,
                            "title_url": title_url,
                            "description": description
                        })
                    return results
                else:
                    print(url)
                    print("No <article> elements found with the specified classes")
                    return []
                
        else:
            return f"Error: Received status code {response.status_code}"
            
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error parsing HTML: {str(e)}"
    
async def translate_news_text(input: str):
    async with Translator() as translator:
        result = await translator.translate(input, src='hu', dest='en')

    return result.text

import asyncio
def add_sentiment_scores(results):

    for item in results:

        text = asyncio.run(translate_news_text(item["description"]))

        inputs = tokenizer(text, max_length=512, return_tensors="pt")
        
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            logits = outputs.logits

        predicted_class = logits.argmax().item()
        item['sentiment_score'] = predicted_class

    return results

def aggregate_sentiment_by_date(results):

    data = {
        'Date': [item['properties'] for item in results],
        'sentiment_score': [item['sentiment_score'] for item in results]
    }
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    # Group by date (day level)
    aggregated = (
        df.groupby(df['Date'].dt.date)['sentiment_score']
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        .reset_index()
    )
    
    aggregated.columns = ['Date', 'sentiment_score']
    aggregated['Date'] = pd.to_datetime(aggregated['Date'])
    return aggregated

def load_news(ticker: str, start_date: str, end_date: str):

    if ticker == "OTP.BD":
        ticker = "otp"

    website_url = f"https://www.portfolio.hu/kereses?q={ticker}&a=&df={start_date}&dt={end_date}&c=&page="
    print(website_url)
    
    max_page = get_page_count(website_url + "1")
    print(max_page)

    all_results = list()

    for i in range(1, max_page + 1):
        # print(i)
        # break
        result = fetch_elements_by_classes(website_url + str(i), TARGET_CLASSES, TARGET_CLASSES_v2)
        all_results.extend(result)
        
        # if i == 5:
        #     break
    with open("news_extracted.txt", "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(f"{item}\n")

    results = add_sentiment_scores(all_results)

    with open("news_with_sentiment.txt", "w", encoding="utf-8") as f:
        for item in results:
            f.write(f"{item}\n")
    return aggregate_sentiment_by_date(results)

def incremental_load_news(ticker:str = "OTP.BD", end_date: str = None):
    conn = sqlite3.connect("../data_pipeline/stock_data.db")

    stock_data = pd.read_sql("SELECT * FROM OTP_BD", conn)
    conn.close()

    start_date = stock_data["Date"].max()

    if start_date < end_date:
        if ticker == "OTP.BD":
            ticker = "otp"

        website_url = f"https://www.portfolio.hu/kereses?q={ticker}&a=&df={start_date}&dt={end_date}&c=&page="
        
        max_page = get_page_count(website_url + "1")

        all_results = list()

        for i in range(1, max_page + 1):
            result = fetch_elements_by_classes(website_url + str(i), TARGET_CLASSES, TARGET_CLASSES_v2)
            all_results.extend(result)
            

        with open("news_extracted_incremental.txt", "w", encoding="utf-8") as f:
            for item in all_results:
                f.write(f"{item}\n")

        results = add_sentiment_scores(all_results)

        with open("news_with_sentiment_incremental.txt", "w", encoding="utf-8") as f:
            for item in results:
                f.write(f"{item}\n")
        return aggregate_sentiment_by_date(results)



if __name__ == "__main__":

    results = load_news("otp")

    print(results)
