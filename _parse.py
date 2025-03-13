# Импорт библиотек для проверки наличия зависимостей
import subprocess
import sys
# Проверка наличия зависиомстей
try:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
except subprocess.CalledProcessError:
    print("Ошибка при установке зависимостей. Убедитесь, что файл requirements.txt присутствует и содержит корректные зависимости.")
    sys.exit(1)

print("Зависимости успешно установлены.")
# Импорт библиотек
from datetime import datetime
import os
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
import re
from urllib.parse import urlparse, urljoin
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from natasha import Doc, Segmenter, NewsNERTagger, NewsEmbedding, MorphVocab
import nltk
base_url = 'https://kolyma.ru/news'
count = 0
nltk.download('punkt_tab')
nltk.download('stopwords')

# Инициализация компонентов Natasha
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)

# Сбор ссылок с новостями
async def gather_links(base_url, session, max_links=100, max_depth=3):
    visited_urls = set()
    urls_to_visit = [(base_url, 0)]  # Хранение URL и глубины поиска
    domain = urlparse(base_url).netloc
    all_links = set()

    while urls_to_visit and len(all_links) < max_links:
        current_url, depth = urls_to_visit.pop(0)

        if current_url in visited_urls or depth > max_depth:
            continue

        visited_urls.add(current_url)

        try:
            html = await fetch(session, current_url)
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                links = soup.find_all('a', href=True)
                for link in links:
                    full_url = urljoin(base_url, link['href'])
                    link_domain = urlparse(full_url).netloc

                    if (
                        full_url not in visited_urls
                        and full_url not in all_links
                        and link_domain == domain
                        and '/news' in full_url 
                    ):
                        all_links.add(full_url)
                        urls_to_visit.append((full_url, depth + 1))
        except Exception as e:
            print(f"Ошибка при обработке страницы {current_url}: {e}")

    return list(all_links)



#Обработка текста из файла
def process_news_data(filename='news_data.csv'):
    processed_data = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            # Пропуск 1 строки
            #next(file, None)
            for line in file:
                if line.strip().startswith("No"):  # Пропуск строки, начинающейся с "No"
                    continue
                if line.strip().startswith("titl"):
                    continue
                try:
                  parts = line.strip().split("|")
                  if len(parts) >= 3: # Проверка title|date|content формата
                      content = parts[2]
                      processed_content = preprocess_text(content)
                      processed_data.append({'title': parts[0], 'date': parts[1], 'content_processed': processed_content})
                  else:
                    print(f"Пропуск строки по причине неправильного формата: {line}")
                except Exception as e:
                  print(f"Произошла ошибка во время обработки строки: {e}, строка: {line}")
    # Обработка исключений для предотвращения ошибок
    except FileNotFoundError:
        print(f"Ошибка! Файл '{filename}' не найден.")
    except Exception as e:
        print(f"Непредсказуемая ошибка: {e}")
    return processed_data
#Предобработка текста
def preprocess_text(text):
    if pd.isna(text):
        return ''
    text = text.lower()  # Приведение к нижнему регистру
    text = re.sub(r'(?<!\w)-(?!\w)|[^\w\s-]', '', text)# Удаление пунктуации, кроме дефисов
    words = word_tokenize(text, language='russian')
    words = [word for word in words if word not in stopwords.words('russian')]  # Удаление стоп-слов
    return ' '.join(words)

# Обработка текста
async def process_text():
    processed_news = process_news_data()

    print("Тексты предобработаны")
    sentences = [item['content_processed'].split() for item in processed_news]
    # Обучение модели word2vec
    print("Обучение модели word2vec")
    model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4) # Инициализация модели
    model.build_vocab(sentences) # Создания словаря
    model.train(sentences, total_examples=model.corpus_count, epochs=10) # Тренировка модели
    model.save('word2vec.model')
    print("Модель word2vec сохранена")

    # Функция для выделения ключевых слов
    def extract_keywords(text, top_n=5):
        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform([text])
        indices = X[0].nonzero()[1]
        scores = zip([vectorizer.get_feature_names_out()[i] for i in indices], [X[0, x] for x in indices])
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_scores[:top_n]]

    
    print("Ключевые слова выделены")

    # Функция для выделения именованных сущностей с помощью Natasha
    def extract_entities(text):
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_ner(ner_tagger)
        entities = []
        for span in doc.spans:
            if span.type == 'PER' or span.type == 'LOC' or span.type == 'ORG':
                entities.append((span.text, span.type))
        return entities
    
    keywords_list = []
    for item in processed_news:
        keywords = extract_keywords(item['content_processed'])
        keywords_list.extend(keywords)
        item['keywords'] = keywords
    
    for item in processed_news:
        item['entities'] = extract_entities(item['content_processed'])

    print("Именованные сущности выделены")
# Отображение топа ключевых имен
    all_keywords = [keyword for item in processed_news for keyword in item['keywords']]
    keyword_counts = Counter(all_keywords)
    print("\nТоп 10 ключевых слов:")
    for keyword, count in keyword_counts.most_common(10):
        print(f"- {keyword}: {count}")
# Отображение топа именнованых сущностей
    all_entities = [entity for item in processed_news for entity in item['entities']]
    entity_counts = Counter(all_entities)
    print("\nТоп 5 именнованых сущностей:")
    for entity, count in entity_counts.most_common(5):
        print(f"- {entity}: {count}")

    # Анализ и визуализация данных

    # Бар-график для популярных ключевых слов
    all_keywords = []
    for item in processed_news:
        all_keywords.extend(item['keywords']) # Использовать ключевые слова из всех элементов

    keywords_freq = Counter(all_keywords).most_common(10)

    plt.figure(figsize=(10, 5))
    plt.bar(*zip(*keywords_freq))
    plt.title('Топ 10 слов')
    plt.xlabel('Слово')
    plt.ylabel('Частота')
    plt.show()

    # Word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_keywords))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    plt.show()


    # Линейный график
    dates = [item['date'] for item in processed_news]

    # Конвертация дат
    try:
        date_objects = [datetime.strptime(date, '%d.%m.%Y') for date in dates]
    except ValueError as e:
        print(f"Ошибка во время преобразования дат: {e}")
        # Обработка
        date_objects = []  
        for date in dates:
            try:
                date_objects.append(datetime.strptime(date, '%d.%m.%Y'))
            except ValueError:
                print(f"Пропускаем неверную дату: {date}")
                date_objects.append(datetime.min)  


# Сортировка дат в хронологическом порядке
    sorted_dates = sorted(date_objects)
    sorted_dates_str = [date.strftime('%d.%m.%Y') for date in sorted_dates] # Преобразование дат в строки

    news_per_day = Counter(sorted_dates_str) # Использование отсортированных дат

# Отрисовка и отображение линейного графика
    plt.figure(figsize=(12, 6))
    plt.plot(list(news_per_day.keys()), list(news_per_day.values()))
    plt.xlabel('Дата')
    plt.ylabel('Кол-во новостей')
    plt.title('Количество новостей за день')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

async def fetch(session, url):
    try:
        
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.text()
    except Exception as e:
        print(f"Ошибка при загрузке {url}: {e}")
        return None


# Получение новостей с сайта
async def fetch_news(session, url):
    global count
    news_data = []
    try:
        html = await fetch(session, url)
        print(f"Загружается страница: {url}")
        if html:
            soup = BeautifulSoup(html, 'html.parser')
            article_body = soup.find('div', class_='article__body') or soup.find('div', itemprop='articleBody')
            date_element = soup.find('meta', itemprop='datePublished')
            title_element = soup.find('h1', itemprop='headline name')

            if title_element:
                span_element = title_element.find('span', class_='masha_index masha_index2')
                if span_element and span_element.next_sibling:
                    title = span_element.next_sibling.strip()
                else:
                    title = title_element.get_text(strip=True)
            else:
                title = "No Title"

            date = date_element['content'] if date_element and 'content' in date_element.attrs else 'No Date'

            if article_body:
                content = article_body.get_text(strip=True)
            else:
                content = "No Content"

            # Date transformation
            match = re.search(r'(\d{1,2})\s+(\w+)\s+(\d{4})', date)
            if match:
                day = match.group(1)
                month_name = match.group(2)
                year = match.group(3)

                month_mapping = {
                    "январь": "01", "февраль": "02", "март": "03", "апрель": "04", "май": "05", "июнь": "06",
                    "июль": "07", "август": "08", "сентябрь": "09", "октябрь": "10", "ноябрь": "11", "декабрь": "12"
                }

                month = month_mapping.get(month_name.lower(), "??") # Handle unknown month names
                formatted_date = f"{day}.{month}.{year}"
            else:
                formatted_date = "No Date"

            count += 1
            news_item = f"{title}|{formatted_date}|{content}"
            news_data.append(news_item)

    except Exception as e:
        print(f"Произошла ошибка: {e}")
        news_data.append("No Title|No Date|No Content")

    return news_data

# Сохранение новостей в CSV
async def save_to_csv(news_data, filename='news_data.csv'):
    df = pd.DataFrame(news_data)
    print("Записываю данные в CSV файл")
    df.to_csv(filename, mode='a', header=False, index=False)  # Запись в файл без заголовка и с добавлением новых данных
    
# Сбор новостей со страниц сайта
async def process_links(links):
    async with aiohttp.ClientSession() as session:
        for url in links:  # Обрабатываем по одной подстранице за раз
            result = await fetch_news(session, url)
            await save_to_csv(result, 'news_data.csv')
    await process_text()  # Сохраняем результаты после обработки каждой страницы

# Пример URL для скачивания новостей
async def to_gather():
    async with aiohttp.ClientSession() as session:
        links = []
        try:
            target_link_count = int(input("Введите желаемое количество ссылок для обработки (не меньше 100): "))
            if target_link_count <= 100:
                raise ValueError
        except ValueError:
            print("Неверный ввод. Используется значение по умолчанию (100).")
            target_link_count = 100
        except Exception as e:
            print(f"Произошла ошибка: {e}. Используется значение по умолчанию (100).")
            target_link_count = 100
        print("Начинаю сбор ссылок (может занять до 1-2 минуты в зависимости от количества ссылок)...")
        page_number = 1
        while len(links) < target_link_count:
            current_url = f"{base_url}/page/{page_number}"
            try:
                new_links = await gather_links(current_url, session)
                if not new_links:
                    print(f"Нет новых ссылок {page_number}.")
                    if len(links) >= target_link_count: # Проверка достигнуто ли минимальное число страниц
                        break
                    else:
                        # Если недостаточно ссылок найдено, прервать цикл
                        print(f"Найдено {len(links)} ссылок.")
                        break
                links.extend(new_links)
                page_number += 1
            except aiohttp.ClientError as e:
                print(f"Ошибка получения страницы {page_number}: {e}")
                break
            except Exception as e:
                print(f"Непредсказуемая ошиька {page_number}: {e}")
                break
        # Call process_links ONLY if the target number of links is reached
        if len(links) >= target_link_count:
            print(f"Найдено {len(links)} ссылок. Начало обработки...")
            await process_links(links)
        else:
            print(f"Целевая ссылка не найдена. Найдено {len(links)} ссылок. Пропуск обработки.")

async def main(base_url):
    try:
        with open('news_data.csv', 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)  # Efficiently count lines

        if line_count > 2:
            while True:
                user_choice = input("news_data.csv существует и там есть данные. Для создания новых данных с нуля (c), а для обработки уже имеющихся данных (p) [c/p]: ")
                if user_choice.lower() == 'c':
                    try:
                        os.remove('news_data.csv')
                        print("news_data.csv удален")
                        await to_gather()
                    except FileNotFoundError:
                        print("news_data.csv не найден. Продолжаю...")
                    except OSError as e:
                        print(f"Ошибка удаления news_data.csv: {e}")
                        sys.exit(1)
                        
                    break
                    
                elif user_choice.lower() == 'p':
                    await process_text()
                    break
                else:
                    print("Неправильный выбор. Выберите 'c' или 'p'.")
        else:
            await to_gather()
    except FileNotFoundError:
        print("news_data.csv не найден. Продожлаю получать новости.")
        async with aiohttp.ClientSession() as session:
            links = await gather_links(base_url, session)
            print(f"Найдено {len(links)} подстраниц.")
            await process_links(links)
    except Exception as e:  # Исключение для других ошибок
        print(f"Произошла ошибка во время обработки news_data.csv: {e}.")
        


# Создаем CSV файл с заголовками перед запуском основного процесса
if not os.path.exists('news_data.csv'):  # Проверка существует ли файл
    with open('news_data.csv', 'w', newline='', encoding='utf-8') as f:
        writer = pd.DataFrame(columns=['title', 'content', 'date']).to_csv(f, index=False)

asyncio.run(main(base_url))
print("Все страницы обработаны.")