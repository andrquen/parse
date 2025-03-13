# Импорт библиотек для проверки наличия зависимостей
import random
import subprocess
import sys
# Проверка наличия зависиомстей
try:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])  # Установка зависимостей из файла requirements.txt
except subprocess.CalledProcessError:
    print("Ошибка при установке зависимостей. Убедитесь, что файл requirements.txt присутствует и содержит корректные зависимости.")
    sys.exit(1)

print("Зависимости успешно установлены.")
# Импорт библиотек
from datetime import datetime
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
base_url = 'https://kolyma.ru/news' # Базовый URL сайта для парсинга
count = 0 # Счетчик обработанных новостей
nltk.download('punkt_tab') # Скачивание необходимых ресурсов для nltk
nltk.download('stopwords')

# Инициализация компонентов Natasha для обработки текста
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)

# Сбор ссылок с новостями с учетом глубины и максимального количества ссылок
async def gather_links(base_url, session, max_links=100, max_depth=3):
    visited_urls = set()  # Множество посещенных URL для избежания дублирования
    urls_to_visit = [(base_url, 0)]  # Очередь URL для посещения с указанием глубины
    domain = urlparse(base_url).netloc  # Домен сайта для ограничения парсинга
    all_links = set()  # Множество всех собранных ссылок

    while urls_to_visit and len(all_links) < max_links:  # Цикл продолжается, пока есть URL для посещения и не достигнут лимит
        current_url, depth = urls_to_visit.pop(0)  # Берем следующий URL из очереди

        if current_url in visited_urls or depth > max_depth:  # Пропускаем уже посещенные URL и превышение глубины
            continue

        visited_urls.add(current_url) # Добавляем текущий URL в посещенные

        try:
            html = await fetch(session, current_url)  # Загружаем HTML код страницы
            if html:
                soup = BeautifulSoup(html, 'html.parser')  # Парсим HTML с помощью BeautifulSoup
                links = soup.find_all('a', href=True)  # Находим все ссылки на странице
                for link in links:
                    full_url = urljoin(base_url, link['href'])  # Формируем полный URL
                    link_domain = urlparse(full_url).netloc  # Извлекаем домен ссылки

                    # Проверяем условия для добавления ссылки:
                    # 1. Ссылка не посещена
                    # 2. Ссылка не в списке всех ссылок
                    # 3. Домен ссылки совпадает с основным доменом
                    # 4. Ссылка содержит '/news' (фильтрация по новостям)
                    if (
                        full_url not in visited_urls
                        and full_url not in all_links
                        and link_domain == domain
                        and '/news' in full_url
                    ):
                        all_links.add(full_url)  # Добавляем ссылку в список всех ссылок
                        urls_to_visit.append((full_url, depth + 1))  # Добавляем ссылку в очередь для посещения с увеличенной глубиной
        except Exception as e:
            print(f"Ошибка при обработке страницы {current_url}: {e}")

    return list(all_links)



#Обработка текста из файла
def process_news_data(filename='news_data.csv'):
    processed_data = [] # Список для хранения обработанных данных
    try:
        with open(filename, 'r', encoding='utf-8') as file: # Открываем файл для чтения с указанием кодировки
            # Пропуск 1 строки
            #next(file, None)
            for line in file: # Читаем файл построчно
                if line.strip().startswith("No"):  # Пропуск строки, начинающейся с "No"
                    continue
                if line.strip().startswith("titl"): # Пропуск строки заголовка, если она существует
                    continue
                try:
                  parts = line.strip().split("|")  # Разделяем строку по разделителю "|"
                  if len(parts) >= 5: # Проверка title|category|tags_string|date|content формата
                      content = parts[4] # Извлекаем контент новости
                      processed_content = preprocess_text(content) # Предобрабатываем контент
                      processed_data.append({'title': parts[0], 'category':parts[1],'tags_string': parts[2],'date': parts[3], 'content_processed': processed_content})  # Добавляем обработанные данные в список
                  else:
                    print(f"Пропуск строки по причине неправильного формата: {line}") # Выводим сообщение о пропуске строки
                except Exception as e:
                  print(f"Произошла ошибка во время обработки строки: {e}, строка: {line}") # Выводим сообщение об ошибке
    # Обработка исключений для предотвращения ошибок
    except FileNotFoundError:
        print(f"Ошибка! Файл '{filename}' не найден.")
    except Exception as e:
        print(f"Непредсказуемая ошибка: {e}")
    return processed_data


#Предобработка текста
def preprocess_text(text):
    if pd.isna(text):  # Проверка на NaN значения
        return ''
    text = text.lower()  # Приведение к нижнему регистру
    text = re.sub(r'(?<!\w)-(?!\w)|[^\w\s-]', '', text) # Удаление пунктуации, кроме дефисов между словами
    words = word_tokenize(text, language='russian')  # Токенизация текста (разделение на слова)
    words = [word for word in words if word not in stopwords.words('russian')]  # Удаление стоп-слов
    return ' '.join(words) # Возвращаем обработанный текст

# Обработка текста: выделение ключевых слов, именованных сущностей, построение моделей и визуализация
async def process_text():
    processed_news = process_news_data() # Получение обработанных данных из файла

    print("Тексты предобработаны")
    sentences = [item['content_processed'].split() for item in processed_news]  # Разделение текста на предложения для word2vec

    # Обучение модели word2vec
    print("Обучение модели word2vec")
    model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)  # Инициализация модели Word2Vec
    model.build_vocab(sentences)  # Построение словаря
    model.train(sentences, total_examples=model.corpus_count, epochs=10)  # Обучение модели
    model.save('word2vec.model') # Сохранение модели
    print("Модель word2vec сохранена")

    # Функция для выделения ключевых слов с помощью TF-IDF
    def extract_keywords(text, top_n=5):
        vectorizer = TfidfVectorizer(max_features=100) # Инициализация TF-IDF векторизатора
        X = vectorizer.fit_transform([text]) # Преобразование текста в вектор
        indices = X[0].nonzero()[1]  # Получение индексов ненулевых элементов
        scores = zip([vectorizer.get_feature_names_out()[i] for i in indices], [X[0, x] for x in indices]) # Создание пар (слово, оценка TF-IDF)
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)  # Сортировка по убыванию оценки TF-IDF
        return [item[0] for item in sorted_scores[:top_n]] # Возвращаем top_n ключевых слов


    print("Ключевые слова выделены")

    # Функция для выделения именованных сущностей с помощью Natasha
    def extract_entities(text):
        doc = Doc(text) # Создание объекта Doc из текста
        doc.segment(segmenter) # Сегментация текста
        doc.tag_ner(ner_tagger) # Разметка именованных сущностей
        entities = [] # Список для хранения именованных сущностей
        for span in doc.spans: # Перебираем все найденные сущности
            if span.type == 'PER' or span.type == 'LOC' or span.type == 'ORG':  # Фильтрация по типам сущностей (PER - Person, LOC - Location, ORG - Organization)
                entities.append((span.text, span.type))  # Добавляем сущность и ее тип в список
        return entities

    keywords_list = [] # Список всех ключевых слов
    for item in processed_news: # Перебираем обработанные новости
        keywords = extract_keywords(item['content_processed']) # Извлекаем ключевые слова из текста
        keywords_list.extend(keywords) # Добавляем ключевые слова в общий список
        item['keywords'] = keywords  # Добавляем ключевые слова в словарь новости

    for item in processed_news: # Перебираем обработанные новости
        item['entities'] = extract_entities(item['content_processed']) # Извлекаем именованные сущности из текста и добавляем в словарь новости

    print("Именованные сущности выделены")
    # Отображение топа ключевых слов
    all_keywords = [keyword for item in processed_news for keyword in item['keywords']] # Создаем список всех ключевых слов
    keyword_counts = Counter(all_keywords) # Подсчитываем частоту каждого ключевого слова
    print("\nТоп 10 ключевых слов:")
    for keyword, count in keyword_counts.most_common(10): # Выводим 10 самых частых ключевых слов
        print(f"- {keyword}: {count}")

    # Отображение топа именованных сущностей
    all_entities = [entity for item in processed_news for entity in item['entities']]  # Создаем список всех именованных сущностей
    entity_counts = Counter(all_entities)  # Подсчитываем частоту каждой именованной сущности
    print("\nТоп 5 именнованых сущностей:")
    for entity, count in entity_counts.most_common(5):  # Выводим 5 самых частых именованных сущностей
        print(f"- {entity}: {count}")

    # Анализ и визуализация данных

    # Бар-график для популярных ключевых слов
    all_keywords = []
    for item in processed_news:
        all_keywords.extend(item['keywords']) # Собираем все ключевые слова

    keywords_freq = Counter(all_keywords).most_common(10)  # Подсчитываем частоту каждого ключевого слова

    plt.figure(figsize=(10, 5))
    plt.bar(*zip(*keywords_freq))  # Строим столбчатую диаграмму
    plt.title('Топ 10 слов')
    plt.xlabel('Слово')
    plt.ylabel('Частота')
    plt.savefig('top_10_words.png') # Save to PNG

    # Word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_keywords))  # Создаем облако слов
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear') # Отображаем облако слов
    plt.axis('off')  # Скрываем оси
    plt.title('Word Cloud')
    plt.savefig('word_cloud.png') # Save to PNG

    # Линейный график количества новостей по датам
    dates = [item['date'] for item in processed_news] # Извлекаем даты из обработанных новостей

    # Конвертация дат в объекты datetime
    try:
        date_objects = [datetime.strptime(date, '%d.%m.%Y') for date in dates]  # Преобразуем строки дат в объекты datetime
    except ValueError as e:
        print(f"Ошибка во время преобразования дат: {e}")
        # Обработка ошибок преобразования дат
        date_objects = []
        for date in dates:
            try:
                date_objects.append(datetime.strptime(date, '%d.%m.%Y'))
            except ValueError:
                print(f"Пропускаем неверную дату: {date}")
                date_objects.append(datetime.min) #  Добавляем минимальную дату для некорректных дат, чтобы не прерывать процесс


    sorted_dates = sorted(date_objects) # Сортируем даты
    sorted_dates_str = [date.strftime('%d.%m.%Y') for date in sorted_dates]  # Преобразуем обратно в строки для корректного отображения на графике

    news_per_day = Counter(sorted_dates_str)  # Подсчитываем количество новостей для каждой даты

    # Отрисовка и отображение линейного графика
    plt.figure(figsize=(12, 6))
    plt.plot(list(news_per_day.keys()), list(news_per_day.values())) # Строим линейный график
    plt.xlabel('Дата')
    plt.ylabel('Кол-во новостей')
    plt.title('Количество новостей за день')
    plt.grid(True) # Добавляем сетку
    plt.xticks(rotation=45) # Поворачиваем метки на оси X для лучшей читаемости
    plt.tight_layout() #  Подгоняем размеры графика
    plt.savefig('news_per_day.png') # Save to PNG

# Функция для загрузки HTML кода страницы
async def fetch(session, url):
    try:
        async with session.get(url) as response: # Отправляем GET запрос
            response.raise_for_status() # Проверяем на ошибки HTTP
            return await response.text()  # Возвращаем текст страницы
    except Exception as e:
        print(f"Ошибка при загрузке {url}: {e}")
        return None
    
# Разделение на категории
url_category_mapping = {
    "/politic": "Власть и политика",
    "/economika": "Экономика и бизнес",
    "/sport": "Спорт",
    "/zdorovie": "Здоровье",
    "/obshestvo": "Общество",
    "/culturenews": "Культура и искусство",
    "/obrazovanie": "Образование",
    "/transport": "Транспорт",
    "/zhkh": "Новости ЖКХ",
    "/proishestvia": "Происшествия",
    "/ecologia": "Экология",
    "/greatday": "День в истории Колыми",
}

# Получение новостей с сайта
async def fetch_news(session, url):

    global count  # Используем глобальную переменную count
    news_data = [] # Список для хранения данных новостей
    try:
        html = await fetch(session, url) # Загружаем HTML код страницы
        print(f"Загружается страница: {url}")
        if html:
            soup = BeautifulSoup(html, 'html.parser') # Парсим HTML
            article_body = soup.find('div', class_='article__body') or soup.find('div', itemprop='articleBody') # Ищем блок с контентом новости
            date_element = soup.find('meta', itemprop='datePublished')  # Ищем элемент с датой публикации
            title_element = soup.find('h1', itemprop='headline name') # Ищем заголово


            if title_element:  # Проверяем, найден ли заголовок
                span_element = title_element.find('span', class_='masha_index masha_index2') # Ищем span с классом masha_index
                if span_element and span_element.next_sibling: # Если span найден и у него есть следующий элемент
                    title = span_element.next_sibling.strip() #  Берем текст следующего элемента как заголовок
                else:
                    title = title_element.get_text(strip=True) # Иначе берем весь текст из title_element
            else:
                title = "No Title" # Если заголовок не найден


            date = date_element['content'] if date_element and 'content' in date_element.attrs else 'No Date' # Извлекаем дату публикации

            if article_body:  # Проверяем, найден ли контент новости
                content = article_body.get_text(strip=True) # Извлекаем текст контента
            else:
                content = "No Content"  # Если контент не найден

            # Преобразование даты в формат дд.мм.гггг
            match = re.search(r'(\d{1,2})\s+(\w+)\s+(\d{4})', date)  # Ищем дату в формате "число месяц год"
            if match:
                day = match.group(1) # Извлекаем день
                month_name = match.group(2) # Извлекаем месяц
                year = match.group(3) #  Извлекаем год
                months = {
                    "январь": "01", "февраль": "02", "март": "03", "апрель": "04", "май": "05", "июнь": "06",
                    "июль": "07", "август": "08", "сентябрь": "09", "октябрь": "10", "ноябрь": "11", "декабрь": "12"
                }
                month = months.get(month_name) # Получаем номер месяца из словаря
                if month:
                    date = f"{day}.{month}.{year}" # Формируем дату в нужном формате

            category = "Разное"  # Категория по умолчанию
            for url_part, cat in url_category_mapping.items():
                if url_part in url:
                    category = cat
                    break
            tag_elements = soup.find_all('li', itemprop='keywords')
            if tag_elements:
                tags = []
                for tag_element in tag_elements:
                    span_element = tag_element.find('span')
                    a_element = span_element.find('a') if span_element else None
                    if a_element:
                        tag_text = a_element.get_text(strip=True)
                        if tag_text:  # Проверка наличия tag_text
                            tags.append(tag_text)
                if tags: # Проверка наличия после фильтрации
                    tags_string = ", ".join(tags)
                else:
                    tags_string = "None"
            else:
                tags_string = "None"
            count += 1 # Увеличиваем счетчик обработанных новостей
            print(f"Новость {count} обработана") # Выводим сообщение об обработке новости
            news_data.append([title,category,tags_string, date, content]) # Добавляем данные новости в список
        return news_data # Возвращаем список данных новостей    
        
        
    except Exception as e:
        print(f"Ошибка при загрузке страницы {url}: {e}")
        return None
    


# ... (previous code remains the same)

async def main():

    max_depth = 3   # Максимальная глубина поиска ссылок
    try:
        target_link_count = int(input("Введите желаемое количество ссылок для обработки (не меньше 100): "))
        if target_link_count < 100:  # Проверка минимального значения
            raise ValueError("Количество ссылок должно быть не менее 100.")
        max_links = target_link_count # Теперь max_links определяется пользователем
    except ValueError as e:
        print(f"Неверный ввод: {e}. Используется значение по умолчанию (100).")
        max_links = 100  # Значение по умолчанию
    except Exception as e:  # обработка других ошибок
        print(f"Произошла ошибка: {e}. Используется значение по умолчанию (100).")
        max_links = 100
    print("Начинается сбор ссылок с новостями...")

    async with aiohttp.ClientSession() as session:  # Создаем клиентскую сессию aiohttp
        links = await gather_links(base_url, session, max_links, max_depth)  # Собираем ссылки на новости
        print(f"Собрано {len(links)} ссылок") # Выводим количество собранных ссылок
        all_news = [] # Список для хранения всех новостей

        if links: # Проверяем, что ссылки найдены
            tasks = [fetch_news(session, link) for link in links]  # Создаем список задач для асинхронной обработки
            for link in links:  # Process links sequentially
                result = await fetch_news(session, link)  # Execute one task
                if result:
                    all_news.extend(result)
                await asyncio.sleep(1)
            results = await asyncio.gather(*tasks) # Запускаем задачи асинхронно

            for result in results: #  Обрабатываем результаты
                if result:
                    all_news.extend(result) # Добавляем данные новостей в общий список
                
            print(f"Всего обработано {len(all_news)} новостей") # Выводим общее количество обработанных новостей
            df = pd.DataFrame(all_news, columns=['title', 'category','tags_string','date', 'content'])  # Создаем DataFrame из списка новостей
            df.to_csv('news_data.csv', sep='|', index=False, encoding='utf-8') #  Сохраняем DataFrame в CSV файл с разделителем "|" и кодировкой UTF-8
            print('Файл news_data.csv создан') #  Выводим сообщение о создании файла

            await process_text()  # Запускаем обработку текста после сбора данных
        else:
            print("Ссылки не найдены. Обработка невозможна.")


if __name__ == '__main__':
    try:
        asyncio.run(main())  # Запускаем асинхронную функцию main
    except Exception as e:
      print(f"Произошла ошибка: {e}") # Выводим сообщение об ошибке, если она произошла

