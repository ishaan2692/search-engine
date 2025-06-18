import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import sqlite3
import hashlib
import time
from io import BytesIO
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Initialize SQLite database
conn = sqlite3.connect('pet_products.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS products
             (id TEXT PRIMARY KEY, title TEXT, description TEXT, 
              price REAL, url TEXT, image BLOB, pet_type TEXT)''')
conn.commit()

# VERIFIED pet product sites with high success rates 
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
    'title': ['h1.product-title', 'h1.product-name', 'h1.title', '[itemprop="name"]', 'h1'],
    'description': ['.product-description', '.details', '[itemprop="description"]', '.product-detail-description', '.description'],
    'price': ['.price', '.product-price', '.pricing', '[itemprop="price"]', '.price-value'],
    'image': ['img.product-image', 'img.primary-image', '[itemprop="image"]', 'img.main-image']
}

# Browser-mimicking headers to avoid blocks 
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
]

# Streamlit app
st.set_page_config(page_title="PawSearch", layout="wide")
st.title("🐾 Pet Product Search Engine")

# Functions
def safe_get(soup, selectors):
    for selector in selectors:
        try:
            element = soup.select_one(selector)
            if element:
                if 'img' in selector:
                    return element.get('src', '')
                return element.get_text(strip=True)
        except:
            continue
    return ''

def get_random_headers():
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Referer': 'https://www.google.com/'
    }

def crawl_site(url, depth=1):
    visited = set()
    to_visit = [url]
    product_links = []
    
    while to_visit and depth > 0:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue
            
        try:
            headers = get_random_headers()
            response = requests.get(current_url, headers=headers, timeout=15)  # Increased timeout
            soup = BeautifulSoup(response.text, 'html.parser')
            visited.add(current_url)
            
            # Find product links using broad patterns 
            for link in soup.find_all('a', href=True):
                href = link['href']
                if not href or href.startswith('javascript:') or href.startswith('mailto:'):
                    continue
                    
                full_url = requests.compat.urljoin(current_url, href)
                
                # Match multiple URL patterns seen in pet sites
                if any(pat in full_url for pat in ['/b/', '/dp/', '/product/', '/p/', '/shop/', '-food', '-toys', '/category/', 'item']):
                    if full_url not in product_links and "customer-reviews" not in full_url:
                        product_links.append(full_url)
                elif depth > 1 and full_url.startswith(url) and '#' not in full_url:
                    to_visit.append(full_url)
            
            depth -= 1
            time.sleep(random.uniform(0.5, 1.5))  # Randomized politeness delay
            
        except Exception as e:
            st.warning(f"Error crawling {current_url}: {str(e)}")
    
    return list(set(product_links))  # Remove duplicates

def scrape_with_retry(url, retries=3):
    for attempt in range(retries):
        try:
            headers = get_random_headers()
            response = requests.get(url, headers=headers, timeout=20)  # Increased timeout
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
                "Dog": ['dog', 'puppy', 'canine', 'k9'],
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
                    img_response = requests.get(product['image'], headers=headers, timeout=10)
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
        
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            wait_time = (attempt + 1) * 5  # Exponential backoff: 5, 10, 15 seconds
            time.sleep(wait_time)
            continue
        except Exception as e:
            st.warning(f"Error scraping {url}: {str(e)}")
            return None
    
    st.warning(f"Failed to scrape {url} after {retries} retries")
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
        all_products = []
        for site in DEMO_SITES:
            with st.spinner(f"Crawling {site}..."):
                try:
                    product_links = crawl_site(site, depth=1)
                    st.info(f"Found {len(product_links)} product links at {site}")
                    
                    if not product_links:
                        continue
                    
                    progress_bar = st.progress(0)
                    scraped_count = 0
                    
                    for i, link in enumerate(product_links):
                        product = scrape_with_retry(link)
                        if product:
                            all_products.append(product)
                            scraped_count += 1
                        
                        progress_bar.progress((i + 1) / len(product_links))
                    
                    st.success(f"Added {scraped_count} products from {site}")
                except Exception as e:
                    st.error(f"Error processing {site}: {str(e)}")
        
        if all_products:
            st.balloons()
            st.success(f"✅ Total added: {len(all_products)} products to database")
        else:
            st.warning("No products were added to the database")
    
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
    st.caption("Note: This demo scrapes real pet product sites. Please respect robots.txt and use responsibly.")

# Main search interface
df, vectors = vectorize_products()

search_query = st.text_input("🔍 Search for pet products:", placeholder="Dog toys, cat food...", key="search_input")

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
                            st.image("https://placekitten.com/150/150", width=150, caption="Placeholder")
                    else:
                        st.image("https://placekitten.com/150/150", width=150, caption="Placeholder")
                
                with col2:
                    title = row['title'] if row['title'] else "Untitled Product"
                    st.subheader(title)
                    
                    if row['price'] and row['price'] > 0:
                        st.metric("Price", f"${row['price']:.2f}")
                    else:
                        st.write("Price: Not available")
                    
                    st.caption(f"**Pet type:** {row['pet_type']}")
                    st.caption(f"**Relevance:** {row['similarity']*100:.1f}%")
                    
                    if row['description']:
                        with st.expander("Description"):
                            st.write(row['description'])
                    else:
                        st.caption("No description available")
                    
                    st.link_button("Visit Product Page", row['url'])
                st.divider()

# Show database table
st.divider()
st.subheader("Product Database")
c.execute("SELECT title, price, pet_type, url FROM products LIMIT 100")
db_data = c.fetchall()

if db_data:
    db_df = pd.DataFrame(db_data, columns=['Title', 'Price', 'Pet Type', 'URL'])
    st.dataframe(db_df, hide_index=True, height=300)
else:
    st.info("Database is empty. Click 'Refresh Database' to populate.")

# Close connection
conn.close()
