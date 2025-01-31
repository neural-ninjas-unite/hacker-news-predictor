import pandas as pd
import psycopg2
from sqlalchemy import create_engine, inspect
import wikipediaapi
import re
from typing import List, Dict
import logging
from tqdm import tqdm
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HackerNewsDataCollector:
    def __init__(self, db_url: str):
        """Initialize the HackerNews data collector with database connection."""
        logger.info("Initializing HackerNews data collector...")
        self.db_url = db_url
        try:
            self.engine = create_engine(db_url)
            # Test connection
            with self.engine.connect() as conn:
                # Check available schemas
                schemas = conn.execute("SELECT schema_name FROM information_schema.schemata").fetchall()
                logger.info(f"Available schemas in database: {[schema[0] for schema in schemas]}")
                logger.info("Successfully connected to database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise
    
    def get_table_names(self) -> List[str]:
        """Get list of available tables in the database."""
        try:
            inspector = inspect(self.engine)
            all_tables = []
            
            # Get tables from all schemas
            schemas = inspector.get_schema_names()
            for schema in schemas:
                if schema != 'information_schema':  # Skip system schema
                    tables = inspector.get_table_names(schema=schema)
                    if tables:
                        logger.info(f"Tables in schema '{schema}': {tables}")
                        all_tables.extend([f"{schema}.{table}" for table in tables])
            
            if not all_tables:
                logger.warning("No tables found in any schema. Please check connection string and permissions.")
            else:
                logger.info(f"All available tables: {all_tables}")
            return all_tables
        except Exception as e:
            logger.error(f"Error inspecting database: {str(e)}")
            raise

    def fetch_hn_data(self) -> pd.DataFrame:
        """Fetch Hacker News data from PostgreSQL database."""
        start_time = time.time()
        logger.info("Starting Hacker News data fetch...")
        
        try:
            query = """
            SELECT 
                title,
                score,
                by as author,
                descendants as num_comments,
                text as comment_text
            FROM hacker_news.items
            WHERE score > 10  -- Increased score threshold
                AND type = 'story'
            LIMIT 50000      -- Limit to 50k records
            """
            
            # Log the query start
            logger.info("Executing database query...")
            df = pd.read_sql(query, self.engine)
            
            # Log statistics about the data
            logger.info(f"Successfully fetched {len(df):,} records")
            logger.info(f"Data statistics:")
            logger.info(f"- Average score: {df['score'].mean():.2f}")
            logger.info(f"- Max score: {df['score'].max()}")
            logger.info(f"- Number of unique authors: {df['author'].nunique():,}")
            logger.info(f"- Number of stories with comments: {df['num_comments'].notna().sum():,}")
            
            # Log the total time taken
            elapsed_time = time.time() - start_time
            logger.info(f"Data collection completed in {elapsed_time:.2f} seconds")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise

class WikipediaDataCollector:
    def __init__(self, language: str = 'en'):
        """Initialize Wikipedia API client."""
        logger.info("Initializing Wikipedia data collector...")
        user_agent = 'HackerNewsPredictor/1.0 (https://github.com/yourusername/hacker-news-predictor; your-email@example.com)'
        self.wiki = wikipediaapi.Wikipedia(
            language=language,
            user_agent=user_agent
        )
        
    def get_articles_by_category(self, category: str, max_articles: int = 100) -> List[str]:
        """Fetch Wikipedia articles from a specific category."""
        start_time = time.time()
        logger.info(f"Starting collection for category: {category}")
        
        category_page = self.wiki.page(f"Category:{category}")
        if not category_page.exists():
            logger.warning(f"Category '{category}' does not exist")
            return []
            
        articles = []
        processed_count = 0
        
        def collect_articles(category_page, depth=0, max_depth=1):
            if depth >= max_depth or len(articles) >= max_articles:
                return
            
            members = list(category_page.categorymembers.values())
            for member in tqdm(members, desc=f"Processing {category} (depth={depth})", leave=False):
                if len(articles) >= max_articles:
                    break
                    
                if member.ns == wikipediaapi.Namespace.MAIN:
                    clean_text = self._clean_text(member.text)
                    if clean_text:
                        articles.append(clean_text)
                        nonlocal processed_count
                        processed_count += 1
                        if processed_count % 5 == 0:
                            elapsed = time.time() - start_time
                            rate = processed_count / elapsed
                            logger.info(f"Processed {processed_count} articles ({rate:.2f} articles/second)")
                        
                elif member.ns == wikipediaapi.Namespace.CATEGORY and depth < max_depth:
                    collect_articles(member, depth + 1, max_depth)
        
        collect_articles(category_page)
        
        # Log summary statistics
        elapsed_time = time.time() - start_time
        logger.info(f"Category '{category}' collection completed:")
        logger.info(f"- Total articles collected: {len(articles):,}")
        logger.info(f"- Average article length: {sum(len(a) for a in articles)/len(articles):.0f} characters")
        logger.info(f"- Time taken: {elapsed_time:.2f} seconds")
        
        return articles
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean Wikipedia article text."""
        # Remove references, URLs, and special characters
        text = re.sub(r'\[\d+\]', '', text)  # Remove reference numbers
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
        return text.strip()

def get_training_data(db_url: str, wiki_categories: List[str]) -> Dict[str, pd.DataFrame]:
    """Collect both HackerNews and Wikipedia training data."""
    start_time = time.time()
    logger.info("=== Starting Data Collection Process ===")
    
    # Collect HackerNews data
    logger.info("Step 1: Collecting Hacker News data...")
    hn_collector = HackerNewsDataCollector(db_url)
    hn_data = hn_collector.fetch_hn_data()
    
    # Collect Wikipedia data
    logger.info("Step 2: Collecting Wikipedia data...")
    wiki_collector = WikipediaDataCollector()
    wiki_texts = []
    
    for idx, category in enumerate(wiki_categories, 1):
        logger.info(f"Processing category {idx}/{len(wiki_categories)}: {category}")
        category_texts = wiki_collector.get_articles_by_category(category)
        wiki_texts.extend(category_texts)
        logger.info(f"Total articles collected so far: {len(wiki_texts):,}")
    
    wiki_data = pd.DataFrame({'text': wiki_texts})
    
    # Log summary statistics
    total_time = time.time() - start_time
    logger.info("\n=== Data Collection Summary ===")
    logger.info(f"Total time taken: {total_time:.2f} seconds")
    logger.info(f"Hacker News records: {len(hn_data):,}")
    logger.info(f"Wikipedia articles: {len(wiki_data):,}")
    logger.info("==============================\n")
    
    return {
        'hacker_news': hn_data,
        'wikipedia': wiki_data
    } 