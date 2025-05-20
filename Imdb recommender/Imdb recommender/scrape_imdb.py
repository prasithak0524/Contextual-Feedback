# scrape_imdb.py
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import pandas as pd
import time

def scrape_imdb_2024():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(service=Service(), options=options)

    movie_names = []
    movie_storylines = []

    url = "https://www.imdb.com/search/title/?year=2024&title_type=feature"
    driver.get(url)
    time.sleep(3)

    movies = driver.find_elements(By.CSS_SELECTOR, ".lister-item.mode-advanced")

    for movie in movies:
        try:
            title = movie.find_element(By.TAG_NAME, "h3").text.split("\n")[0]
            desc = movie.find_element(By.CLASS_NAME, "text-muted").find_elements(By.TAG_NAME, "p")[1].text
            movie_names.append(title)
            movie_storylines.append(desc)
        except:
            continue

    df = pd.DataFrame({"Movie Name": movie_names, "Storyline": movie_storylines})
    df.to_csv("data/imdb_2024_movies.csv", index=False)
    driver.quit()
    print("Data saved to imdb_2024_movies.csv")

if __name__ == "__main__":
    scrape_imdb_2024()
