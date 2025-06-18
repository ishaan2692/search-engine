import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import sqlite3
import hashlib
import time
from io import BytesIO
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize SQLite database
conn = sqlite3.connect('pet_products.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS products
             (id TEXT PRIMARY KEY, title TEXT, description TEXT, 
              price REAL, url TEXT, image BLOB, pet_type TEXT)''')
conn.commit()

# VERIFIED pet product sites with high success rates :cite[1]:cite[6]:cite[9]
DEMO_SITES = [
    "https://www.chewy.com/b/dog-food-387",
    "https://www.chewy.com/b/cat-toys-335",
    "https://www.petsmart.com/dog/food/",
    "https://www.petco.com/shop/en/petcostore/category/dog/dog-food"
]

# Initialize vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Enhanced product schema with fallback selectors
PRODUCT_SCHEMA = {
    'title': ['h1.product-title', 'h1.product-name', 'h1.title', '[itemprop="name"]'],
    'description': ['.product-description', '.details', '[itemprop="description"]', '.product-detail-description'],
    'price': ['.price', '.product-price', '.pricing', '[itemprop="price"]'],
    'image': ['img.product-image', 'img.primary-image', '[itemprop="image"]']
}

# Browser-mimicking headers to avoid blocks :cite[4]:cite[9]
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
}

# Streamlit app
st.set_page_config(page_title="PetPaws Search", layout="wide")
st.title("🐾 PetPaws - Pet Product Search Engine")

# Functions
def safe_get(soup, selectors):
    for selector in selectors:
        element = soup.select_one(selector)
        if element:
            if 'img' in selector:
                return element.get('src', '')
            return element.get_text(strip=True)
    return ''

def crawl_site(url, depth=2):
    visited = set()
    to_visit = [url]
    product_links = []
    
    while to_visit and depth > 0:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue
            
        try:
            response = requests.get(current_url, headers=DEFAULT_HEADERS, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            visited.add(current_url)
            
            # Find product links using broad patterns :cite[1]:cite[6]
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = requests.compat.urljoin(current_url, href)
                
                # Match multiple URL patterns seen in pet sites
                if any(pat in full_url for pat in ['/b/', '/dp/', '/product/', '/p/', '/shop/', '-food', '-toys', '/category/']):
                    if full_url not in product_links and "customer-reviews" not in full_url:
                        product_links.append(full_url)
                elif depth > 1 and full_url.startswith(url):
                    to_visit.append(full_url)
            
            depth -= 1
            time.sleep(1)  # Increased politeness delay
            
        except Exception as e:
            st.warning(f"Error crawling {current_url}: {str(e)}")
    
    return list(set(product_links))  # Remove duplicates

def scrape_product(url):
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        product = {'url': url}
        for key, selectors in PRODUCT_SCHEMA.items():
            product[key] = safe_get(soup, selectors)
        
        # Price cleaning with currency awareness
        if product['price']:
            match = re.search(r'(\d+[\.,]\d{1,2})', product['price'])
            if match:
                product['price'] = float(match.group().replace(',', ''))
        
        # Enhanced pet type detection
        pet_type = "Other"
        pet_keywords = {
            "Dog": ['dog', 'puppy', 'canine'],
            "Cat": ['cat', 'kitten', 'feline'],
            "Fish": ['fish', 'aquarium', 'aquatic'],
            "Bird": ['bird', 'parrot', 'avian']
        }
        for animal, terms in pet_keywords.items():
            if any(t in url.lower() or (product.get('title') and t in product['title'].lower()) for t in terms):
                pet_type = animal
                break
        product['pet_type'] = pet_type
        
        # Image handling with error resilience
        img_data = b''
        if product['image'] and product['image'].startswith('http'):
            try:
                img_response = requests.get(product['image'], timeout=5)
                if img_response.status_code == 200:
                    img_data = img_response.content
            except:
                pass
        
        # Create unique ID
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        
        # Save to DB (ignore duplicates)
        c.execute('''INSERT OR IGNORE INTO products VALUES (?,?,?,?,?,?,?)''',
                  (url_hash, product.get('title', ''), product.get('description', ''),
                   product.get('price', 0), url, img_data, pet_type))
        conn.commit()
        
        return product
    
    except Exception as e:
        st.warning(f"Error scraping {url}: {str(e)}")
        return None

# ... (rest of functions unchanged from original)

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    
    if st.button("🔄 Refresh Database (Fixed)"):
        with st.spinner("Crawling sites with improved logic..."):
            all_products = []
            for site in DEMO_SITES:
                product_links = crawl_site(site, depth=2)
                st.info(f"Found {len(product_links)} product links at {site}")
                for i, link in enumerate(product_links):
                    product = scrape_product(link)
                    if product:
                        all_products.append(product)
                    if i % 5 == 0:  # Progress tracking
                        st.sidebar.text(f"Scraped: {i+1}/{len(product_links)} @ {site}")
            st.success(f"Added {len(all_products)} products to database")

# ... (rest of UI unchanged)
