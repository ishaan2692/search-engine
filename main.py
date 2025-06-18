import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import time
import hashlib
import sqlite3
from io import BytesIO
from PIL import Image

# Initialize SQLite database
conn = sqlite3.connect('pet_products.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS products
             (id TEXT PRIMARY KEY, title TEXT, description TEXT, 
              price REAL, url TEXT, image BLOB, pet_type TEXT)''')
conn.commit()

# Sample pet product sites for crawling (safe for demo)
DEMO_SITES = [
    "https://www.petsmart.com/dog/food/",
    "https://www.petsmart.com/dog/toys/",
    "https://www.petsmart.com/cat/food/",
    "https://www.petsmart.com/cat/toys/"
]

# Initialize vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Product schema for scraping
PRODUCT_SCHEMA = {
    'title': ['h1', 'product-title', 'product-name'],
    'description': ['.product-description', '[itemprop="description"]', '.detail-content'],
    'price': ['.price', '.product-price', '[itemprop="price"]'],
    'image': ['img.product-image', '[itemprop="image"]']
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

def crawl_site(url, depth=1):
    visited = set()
    to_visit = [url]
    product_links = []
    
    while to_visit and depth > 0:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue
            
        try:
            response = requests.get(current_url, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            visited.add(current_url)
            
            # Find product links
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = requests.compat.urljoin(current_url, href)
                
                if '/product/' in full_url or '/p/' in full_url:
                    if full_url not in product_links:
                        product_links.append(full_url)
                elif depth > 1 and full_url.startswith(url):
                    to_visit.append(full_url)
            
            depth -= 1
            time.sleep(0.5)  # Be polite
            
        except Exception as e:
            st.warning(f"Error crawling {current_url}: {str(e)}")
    
    return product_links

def scrape_product(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        product = {'url': url}
        for key, selectors in PRODUCT_SCHEMA.items():
            product[key] = safe_get(soup, selectors)
        
        # Clean price
        if product['price']:
            match = re.search(r'[\d,.]+', product['price'])
            if match:
                product['price'] = float(match.group().replace(',', ''))
        
        # Detect pet type
        pet_type = "Other"
        if any(t in url.lower() for t in ['dog', 'puppy', 'canine']):
            pet_type = "Dog"
        elif any(t in url.lower() for t in ['cat', 'kitten', 'feline']):
            pet_type = "Cat"
        product['pet_type'] = pet_type
        
        # Download image
        img_data = b''
        if product['image'] and product['image'].startswith('http'):
            try:
                img_response = requests.get(product['image'], timeout=5)
                img_data = img_response.content
            except:
                pass
        
        # Create unique ID
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        
        # Save to DB
        c.execute('''REPLACE INTO products VALUES (?,?,?,?,?,?,?)''',
                  (url_hash, product['title'], product['description'], 
                   product.get('price', 0), url, img_data, pet_type))
        conn.commit()
        
        return product
    
    except Exception as e:
        st.warning(f"Error scraping {url}: {str(e)}")
        return None

def vectorize_products():
    c.execute("SELECT * FROM products")
    products = c.fetchall()
    
    if not products:
        return None, None
    
    df = pd.DataFrame(products, columns=['id','title','description','price','url','image','pet_type'])
    
    # Create text corpus
    df['text'] = df['title'].fillna('') + " " + df['description'].fillna('')
    
    # Vectorize
    vectorizer.fit(df['text'])
    vectors = vectorizer.transform(df['text'])
    
    return df, vectors

def search_products(query, df, vectors, top_k=5):
    if df is None or vectors is None:
        return pd.DataFrame()
    
    # Vectorize query
    query_vec = vectorizer.transform([query])
    
    # Calculate similarity
    similarities = cosine_similarity(query_vec, vectors).flatten()
    
    # Get top results
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = df.iloc[top_indices].copy()
    results['similarity'] = similarities[top_indices]
    
    return results

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    
    if st.button("🔄 Refresh Database"):
        with st.spinner("Crawling sites..."):
            all_products = []
            for site in DEMO_SITES:
                product_links = crawl_site(site, depth=1)
                for link in product_links:
                    product = scrape_product(link)
                    if product:
                        all_products.append(product)
            st.success(f"Added {len(all_products)} products to database")
    
    if st.button("🧹 Clear Database"):
        c.execute("DELETE FROM products")
        conn.commit()
        st.success("Database cleared!")
    
    st.divider()
    st.info("Current database stats:")
    c.execute("SELECT COUNT(*) FROM products")
    count = c.fetchone()[0]
    st.write(f"📦 Products: {count}")
    
    if count > 0:
        c.execute("SELECT pet_type, COUNT(*) FROM products GROUP BY pet_type")
        for row in c.fetchall():
            st.write(f"- {row[0]}: {row[1]}")
    
    st.divider()
    st.caption("Note: This demo uses sample pet product sites. Actual scraping may be limited by website policies.")

# Main search interface
df, vectors = vectorize_products()

search_query = st.text_input("🔍 Search for pet products:", placeholder="Dog toys, cat food...")

if st.button("Search") or search_query:
    if not search_query.strip():
        st.warning("Please enter a search query")
        st.stop()
    
    with st.spinner("Finding best products..."):
        results = search_products(search_query, df, vectors, top_k=10)
        
        if results.empty:
            st.warning("No matching products found. Try refreshing the database.")
            st.stop()
            
        st.subheader(f"Top {len(results)} Results for '{search_query}'")
        
        for _, row in results.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    if row['image']:
                        try:
                            img = Image.open(BytesIO(row['image']))
                            st.image(img, width=150)
                        except:
                            st.image("https://placekitten.com/150/150", width=150)
                    else:
                        st.image("https://placekitten.com/150/150", width=150)
                
                with col2:
                    st.subheader(row['title'])
                    
                    if row['price']:
                        st.metric("Price", f"${row['price']:.2f}")
                    else:
                        st.write("Price not available")
                    
                    st.caption(f"**Pet type:** {row['pet_type']}")
                    st.caption(f"**Similarity:** {row['similarity']*100:.1f}%")
                    
                    with st.expander("Description"):
                        st.write(row['description'] or "No description available")
                    
                    st.link_button("Visit Product Page", row['url'])

# Show database table
st.divider()
st.subheader("Product Database")
c.execute("SELECT title, price, pet_type, url FROM products")
db_data = c.fetchall()

if db_data:
    db_df = pd.DataFrame(db_data, columns=['Title', 'Price', 'Pet Type', 'URL'])
    st.dataframe(db_df, hide_index=True)
else:
    st.info("Database is empty. Click 'Refresh Database' to populate.")

# Close connection
conn.close()
