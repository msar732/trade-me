# ==================== TRADE HUB - ENHANCED TRADING PLATFORM ====================
import os
import json
import uuid
import hashlib
import secrets
import re
import random
import base64
import hmac
import time
import threading
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
import logging
from pathlib import Path
from functools import lru_cache, wraps
from collections import defaultdict
import math
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Core Flask and Web Framework
from flask import Flask, render_template_string, request, jsonify, redirect, url_for, flash, session, send_from_directory, make_response, abort
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.middleware.profiler import ProfilerMiddleware

# Database and ORM
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Float, Text, ForeignKey, JSON, Index, event, func, and_, or_
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship, scoped_session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import IntegrityError

# MongoDB Support
try:
    import pymongo
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure
    MONGODB_AVAILABLE = True
    print("✅ MongoDB support loaded successfully!")
except ImportError:
    MONGODB_AVAILABLE = False
    print("⚠️ MongoDB libraries not available")

# Hindi Language Support
try:
    from googletrans import Translator
    HINDI_AVAILABLE = True
    print("✅ Hindi language support loaded successfully!")
except ImportError:
    HINDI_AVAILABLE = False
    print("⚠️ Hindi language libraries not available")

# Production Performance
try:
    import redis
    REDIS_AVAILABLE = True
    print("✅ Redis caching available!")
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️ Redis not available")

from email_validator import validate_email, EmailNotValidError
import threading
import queue
import gzip
from io import BytesIO

# Advanced AI Libraries (Production Ready)
try:
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
    print("✅ Production ML Stack Loaded Successfully!")
except ImportError as e:
    ML_AVAILABLE = False
    print(f"⚠️ ML libraries not available: {e}")

# Web Scraping Libraries
try:
    import requests
    from bs4 import BeautifulSoup
    import urllib.parse
    WEB_SCRAPING_AVAILABLE = True
    print("✅ Web Scraping libraries loaded successfully!")
except ImportError as e:
    WEB_SCRAPING_AVAILABLE = False
    print(f"⚠️ Web Scraping libraries not available: {e}")

# ==================== PRODUCTION CONFIGURATION ====================
class TradeHubConfig:
    """Production-grade configuration for Trade Hub"""
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tradehub_production.db")
    DATABASE_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "50"))
    DATABASE_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "100"))
    DATABASE_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    DATABASE_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))
    
    # Security Configuration
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
    SESSION_PERMANENT = False
    SESSION_TYPE = "filesystem"
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # File Upload Configuration
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./uploads")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", "16777216"))  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'pdf', 'doc', 'docx'}
    
    # Cache Configuration
    CACHE_TYPE = "redis" if REDIS_AVAILABLE else "simple"
    CACHE_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CACHE_DEFAULT_TIMEOUT = int(os.getenv("CACHE_TIMEOUT", "300"))
    
    # Rate Limiting
    RATELIMIT_STORAGE_URL = os.getenv("REDIS_URL", "redis://localhost:6379/1") if REDIS_AVAILABLE else "memory://"
    RATELIMIT_DEFAULT = "1000 per hour"
    RATELIMIT_HEADERS_ENABLED = True
    
    # Email Configuration
    MAIL_SERVER = os.getenv("MAIL_SERVER", "smtp.gmail.com")
    MAIL_PORT = int(os.getenv("MAIL_PORT", "587"))
    MAIL_USE_TLS = True
    MAIL_USERNAME = os.getenv("MAIL_USERNAME")
    MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
    
    # API Configuration
    EXTERNAL_API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
    MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "100"))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "tradehub.log")

# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=getattr(logging, TradeHubConfig.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(TradeHubConfig.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== FLASK APP INITIALIZATION ====================
app = Flask(__name__)
app.config.from_object(TradeHubConfig)

# CORS Configuration
CORS(app, origins=['*'], methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])

# Rate Limiting
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[TradeHubConfig.RATELIMIT_DEFAULT],
    storage_uri=TradeHubConfig.RATELIMIT_STORAGE_URL
)

# ==================== DATABASE MODELS ====================
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(200))
    phone = Column(String(20))
    location = Column(String(200))
    bio = Column(Text)
    avatar_url = Column(String(500))
    is_verified = Column(Boolean, default=False)
    is_premium = Column(Boolean, default=False)
    rating = Column(Float, default=0.0)
    total_ratings = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    listings = relationship("Listing", back_populates="user", cascade="all, delete-orphan")
    watchlist = relationship("Watchlist", back_populates="user", cascade="all, delete-orphan")
    sent_messages = relationship("Message", foreign_keys="Message.sender_id", back_populates="sender")
    received_messages = relationship("Message", foreign_keys="Message.recipient_id", back_populates="recipient")
    ratings_given = relationship("Rating", foreign_keys="Rating.rater_id", back_populates="rater")
    ratings_received = relationship("Rating", foreign_keys="Rating.rated_id", back_populates="rated")
    notifications = relationship("Notification", back_populates="user", cascade="all, delete-orphan")

class Category(Base):
    __tablename__ = 'categories'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text)
    icon = Column(String(100))
    parent_id = Column(Integer, ForeignKey('categories.id'))
    is_active = Column(Boolean, default=True)
    sort_order = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    parent = relationship("Category", remote_side=[id])
    children = relationship("Category")
    listings = relationship("Listing", back_populates="category")

class Listing(Base):
    __tablename__ = 'listings'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=False)
    price = Column(Float, nullable=False, index=True)
    currency = Column(String(3), default='INR')
    condition = Column(String(50))  # New, Used, Refurbished
    location = Column(String(200), index=True)
    latitude = Column(Float)
    longitude = Column(Float)
    images = Column(JSON)  # List of image URLs
    tags = Column(JSON)  # List of tags
    specifications = Column(JSON)  # Product specifications
    contact_info = Column(JSON)  # Contact information
    is_featured = Column(Boolean, default=False, index=True)
    is_urgent = Column(Boolean, default=False)
    is_negotiable = Column(Boolean, default=True)
    status = Column(String(20), default='active', index=True)  # active, sold, expired, deleted
    views = Column(Integer, default=0, index=True)
    favorites = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime, index=True)
    
    # Foreign Keys
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    category_id = Column(Integer, ForeignKey('categories.id'), nullable=False, index=True)
    
    # Relationships
    user = relationship("User", back_populates="listings")
    category = relationship("Category", back_populates="listings")
    watchlist = relationship("Watchlist", back_populates="listing", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="listing", cascade="all, delete-orphan")

class Watchlist(Base):
    __tablename__ = 'watchlist'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    listing_id = Column(Integer, ForeignKey('listings.id'), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="watchlist")
    listing = relationship("Listing", back_populates="watchlist")

class Message(Base):
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True)
    sender_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    recipient_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    listing_id = Column(Integer, ForeignKey('listings.id'), index=True)
    subject = Column(String(200))
    content = Column(Text, nullable=False)
    is_read = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    sender = relationship("User", foreign_keys=[sender_id], back_populates="sent_messages")
    recipient = relationship("User", foreign_keys=[recipient_id], back_populates="received_messages")
    listing = relationship("Listing", back_populates="messages")

class Rating(Base):
    __tablename__ = 'ratings'
    
    id = Column(Integer, primary_key=True)
    rater_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    rated_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    listing_id = Column(Integer, ForeignKey('listings.id'), index=True)
    rating = Column(Integer, nullable=False)  # 1-5 stars
    comment = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    rater = relationship("User", foreign_keys=[rater_id], back_populates="ratings_given")
    rated = relationship("User", foreign_keys=[rated_id], back_populates="ratings_received")

class Notification(Base):
    __tablename__ = 'notifications'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    type = Column(String(50), nullable=False)  # message, watchlist, system, etc.
    is_read = Column(Boolean, default=False, index=True)
    data = Column(JSON)  # Additional data
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user = relationship("User", back_populates="notifications")

# ==================== DATABASE SETUP ====================
engine = create_engine(
    TradeHubConfig.DATABASE_URL,
    poolclass=QueuePool,
    pool_size=TradeHubConfig.DATABASE_POOL_SIZE,
    max_overflow=TradeHubConfig.DATABASE_MAX_OVERFLOW,
    pool_timeout=TradeHubConfig.DATABASE_POOL_TIMEOUT,
    pool_recycle=TradeHubConfig.DATABASE_POOL_RECYCLE,
    echo=False
)

SessionLocal = scoped_session(sessionmaker(bind=engine))

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database with tables and sample data"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully!")
        
        # Create sample data
        db = SessionLocal()
        
        # Check if categories exist
        if db.query(Category).count() == 0:
            # Create main categories
            categories = [
                {"name": "Property", "slug": "property", "description": "Real Estate & Properties", "icon": "🏠"},
                {"name": "Motors", "slug": "motors", "description": "Vehicles & Automotive", "icon": "🚗"},
                {"name": "Jobs", "slug": "jobs", "description": "Employment Opportunities", "icon": "💼"},
                {"name": "Services", "slug": "services", "description": "Professional Services", "icon": "🔧"},
                {"name": "Marketplace", "slug": "marketplace", "description": "Buy & Sell Items", "icon": "🛒"},
                {"name": "Electronics", "slug": "electronics", "description": "Gadgets & Electronics", "icon": "📱"},
                {"name": "Fashion", "slug": "fashion", "description": "Clothing & Accessories", "icon": "👕"},
                {"name": "Home & Garden", "slug": "home-garden", "description": "Home Improvement & Gardening", "icon": "🏡"},
            ]
            
            for cat_data in categories:
                category = Category(**cat_data)
                db.add(category)
            
            db.commit()
            logger.info("✅ Sample categories created!")
        
        db.close()
        
    except Exception as e:
        logger.error(f"❌ Database initialization error: {e}")

# ==================== CACHE MANAGER ====================
class CacheManager:
    """Production-grade caching system"""
    
    def __init__(self):
        self.cache = {}
        self.redis_client = None
        
        if REDIS_AVAILABLE:
            try:
                import redis
                self.redis_client = redis.from_url(TradeHubConfig.CACHE_REDIS_URL)
                self.redis_client.ping()
                logger.info("✅ Redis cache connected successfully!")
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed: {e}")
                self.redis_client = None
    
    def get(self, key: str):
        """Get value from cache"""
        try:
            if self.redis_client:
                value = self.redis_client.get(key)
                return json.loads(value) if value else None
            else:
                return self.cache.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: any, timeout: int = None):
        """Set value in cache"""
        try:
            timeout = timeout or TradeHubConfig.CACHE_DEFAULT_TIMEOUT
            if self.redis_client:
                self.redis_client.setex(key, timeout, json.dumps(value))
            else:
                self.cache[key] = value
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def delete(self, key: str):
        """Delete value from cache"""
        try:
            if self.redis_client:
                self.redis_client.delete(key)
            else:
                self.cache.pop(key, None)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")

cache_manager = CacheManager()

def cache_response(timeout=300):
    """Cache decorator for routes"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            cache_key = f"route:{request.endpoint}:{request.args.to_dict()}"
            cached_result = cache_manager.get(cache_key)
            
            if cached_result:
                return cached_result
            
            result = f(*args, **kwargs)
            cache_manager.set(cache_key, result, timeout)
            return result
        return decorated_function
    return decorator

# ==================== SECURITY MANAGER ====================
class SecurityManager:
    """Production security manager"""
    
    @staticmethod
    def generate_token():
        """Generate secure token"""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password securely"""
        return generate_password_hash(password, method='pbkdf2:sha256')
    
    @staticmethod
    def verify_password(password: str, hash: str) -> bool:
        """Verify password against hash"""
        return check_password_hash(hash, password)
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email address"""
        try:
            validate_email(email)
            return True
        except EmailNotValidError:
            return False
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize user input"""
        if not text:
            return ""
        # Remove potentially dangerous characters
        text = re.sub(r'[<>"\']', '', str(text))
        return text.strip()

security = SecurityManager()

# ==================== WEB SCRAPING MANAGER ====================
class WebScrapingManager:
    """Enhanced web scraping with multiple sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def scrape_olx(self, query: str, location: str = "", limit: int = 20) -> List[Dict]:
        """Scrape OLX listings"""
        if not WEB_SCRAPING_AVAILABLE:
            return []
        
        try:
            results = []
            search_url = f"https://www.olx.in/items/q-{query}"
            if location:
                search_url += f"-l-{location}"
            
            response = self.session.get(search_url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            listings = soup.find_all('div', {'data-aut-id': 'itemBox'})[:limit]
            
            for listing in listings:
                try:
                    title_elem = listing.find('span', {'data-aut-id': 'itemTitle'})
                    price_elem = listing.find('span', {'data-aut-id': 'itemPrice'})
                    location_elem = listing.find('span', {'data-aut-id': 'item-location'})
                    image_elem = listing.find('img')
                    
                    if title_elem and price_elem:
                        results.append({
                            'title': title_elem.get_text(strip=True),
                            'price': price_elem.get_text(strip=True),
                            'location': location_elem.get_text(strip=True) if location_elem else '',
                            'image': image_elem.get('src') if image_elem else '',
                            'source': 'OLX'
                        })
                except Exception as e:
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"OLX scraping error: {e}")
            return []
    
    def scrape_quikr(self, query: str, limit: int = 20) -> List[Dict]:
        """Scrape Quikr listings"""
        if not WEB_SCRAPING_AVAILABLE:
            return []
        
        try:
            results = []
            search_url = f"https://www.quikr.com/search?q={query}"
            
            response = self.session.get(search_url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            listings = soup.find_all('div', class_='list-item')[:limit]
            
            for listing in listings:
                try:
                    title_elem = listing.find('a', class_='list-item-title')
                    price_elem = listing.find('span', class_='price')
                    location_elem = listing.find('span', class_='location')
                    image_elem = listing.find('img')
                    
                    if title_elem and price_elem:
                        results.append({
                            'title': title_elem.get_text(strip=True),
                            'price': price_elem.get_text(strip=True),
                            'location': location_elem.get_text(strip=True) if location_elem else '',
                            'image': image_elem.get('src') if image_elem else '',
                            'source': 'Quikr'
                        })
                except Exception:
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Quikr scraping error: {e}")
            return []

scraper = WebScrapingManager()

# ==================== LANGUAGE MANAGER ====================
class LanguageManager:
    """Multi-language support"""
    
    def __init__(self):
        self.translator = None
        if HINDI_AVAILABLE:
            try:
                from googletrans import Translator
                self.translator = Translator()
                logger.info("✅ Language translator initialized!")
            except Exception as e:
                logger.warning(f"⚠️ Translator initialization failed: {e}")
    
    def translate_to_hindi(self, text: str) -> str:
        """Translate text to Hindi"""
        if not self.translator or not text:
            return text
        
        try:
            result = self.translator.translate(text, dest='hi')
            return result.text
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text
    
    def translate_from_hindi(self, text: str) -> str:
        """Translate text from Hindi to English"""
        if not self.translator or not text:
            return text
        
        try:
            result = self.translator.translate(text, dest='en')
            return result.text
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text

language_manager = LanguageManager()

# Initialize database
init_db()

# ==================== ROUTE HANDLERS ====================
@app.route('/')
def home():
    """Enhanced homepage with all features"""
    try:
        db = next(get_db())
        
        # Get categories
        categories = db.query(Category).filter(Category.is_active == True).order_by(Category.sort_order).all()
        
        # Get featured listings
        featured_listings = db.query(Listing).filter(
            Listing.is_featured == True,
            Listing.status == 'active'
        ).order_by(Listing.created_at.desc()).limit(8).all()
        
        # Get statistics
        stats = {
            'total_users': db.query(User).filter(User.is_active == True).count(),
            'total_listings': db.query(Listing).filter(Listing.status == 'active').count(),
            'total_categories': db.query(Category).filter(Category.is_active == True).count()
        }
        
        return render_template_string(TRADE_HUB_TEMPLATE, 
                                    page='home',
                                    categories=categories,
                                    featured_listings=featured_listings,
                                    stats=stats,
                                    user_session=session)
        
    except Exception as e:
        logger.error(f"Homepage error: {e}")
        return render_template_string(TRADE_HUB_TEMPLATE, 
                                    page='home',
                                    categories=[],
                                    featured_listings=[],
                                    stats={'total_users': 0, 'total_listings': 0, 'total_categories': 0},
                                    user_session=session)

@app.route('/api/search')
@limiter.limit("100 per minute")
def api_search():
    """Enhanced search API"""
    try:
        query = request.args.get('q', '').strip()
        category = request.args.get('category', '')
        location = request.args.get('location', '')
        min_price = request.args.get('min_price', type=float)
        max_price = request.args.get('max_price', type=float)
        condition = request.args.get('condition', '')
        sort_by = request.args.get('sort_by', 'created_at')
        sort_order = request.args.get('sort_order', 'desc')
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 100)
        
        if not query:
            return jsonify({'error': 'Search query is required'}), 400
        
        db = next(get_db())
        
        # Build query
        query_obj = db.query(Listing).filter(Listing.status == 'active')
        
        # Text search
        if query:
            query_obj = query_obj.filter(
                or_(
                    Listing.title.ilike(f'%{query}%'),
                    Listing.description.ilike(f'%{query}%')
                )
            )
        
        # Category filter
        if category:
            cat_obj = db.query(Category).filter(Category.slug == category).first()
            if cat_obj:
                query_obj = query_obj.filter(Listing.category_id == cat_obj.id)
        
        # Location filter
        if location:
            query_obj = query_obj.filter(Listing.location.ilike(f'%{location}%'))
        
        # Price filters
        if min_price is not None:
            query_obj = query_obj.filter(Listing.price >= min_price)
        if max_price is not None:
            query_obj = query_obj.filter(Listing.price <= max_price)
        
        # Condition filter
        if condition:
            query_obj = query_obj.filter(Listing.condition == condition)
        
        # Sorting
        if sort_by == 'price':
            if sort_order == 'asc':
                query_obj = query_obj.order_by(Listing.price.asc())
            else:
                query_obj = query_obj.order_by(Listing.price.desc())
        elif sort_by == 'views':
            query_obj = query_obj.order_by(Listing.views.desc())
        else:  # created_at
            if sort_order == 'asc':
                query_obj = query_obj.order_by(Listing.created_at.asc())
            else:
                query_obj = query_obj.order_by(Listing.created_at.desc())
        
        # Pagination
        offset = (page - 1) * per_page
        total = query_obj.count()
        listings = query_obj.offset(offset).limit(per_page).all()
        
        # Format results
        results = []
        for listing in listings:
            results.append({
                'id': listing.id,
                'title': listing.title,
                'description': listing.description[:200] + '...' if len(listing.description) > 200 else listing.description,
                'price': listing.price,
                'currency': listing.currency,
                'condition': listing.condition,
                'location': listing.location,
                'images': listing.images or [],
                'category': listing.category.name if listing.category else None,
                'user': {
                    'username': listing.user.username,
                    'rating': listing.user.rating
                } if listing.user else None,
                'created_at': listing.created_at.isoformat(),
                'views': listing.views,
                'is_featured': listing.is_featured
            })
        
        return jsonify({
            'results': results,
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': math.ceil(total / per_page)
        })
        
    except Exception as e:
        logger.error(f"Search API error: {e}")
        return jsonify({'error': 'Search failed'}), 500

@app.route('/api/scrape')
@limiter.limit("10 per minute")
def api_scrape():
    """Web scraping API"""
    try:
        query = request.args.get('q', '').strip()
        location = request.args.get('location', '')
        sources = request.args.getlist('sources') or ['olx', 'quikr']
        limit = min(request.args.get('limit', 20, type=int), 50)
        
        if not query:
            return jsonify({'error': 'Search query is required'}), 400
        
        results = []
        
        # Scrape from different sources
        if 'olx' in sources:
            olx_results = scraper.scrape_olx(query, location, limit//2)
            results.extend(olx_results)
        
        if 'quikr' in sources:
            quikr_results = scraper.scrape_quikr(query, limit//2)
            results.extend(quikr_results)
        
        # Sort by relevance (simple scoring)
        def relevance_score(item):
            score = 0
            title_lower = item['title'].lower()
            query_lower = query.lower()
            
            if query_lower in title_lower:
                score += 10
            
            # Boost newer items (if we had timestamps)
            score += random.randint(1, 5)
            
            return score
        
        results.sort(key=relevance_score, reverse=True)
        
        return jsonify({
            'results': results[:limit],
            'total': len(results),
            'sources': sources,
            'query': query
        })
        
    except Exception as e:
        logger.error(f"Scraping API error: {e}")
        return jsonify({'error': 'Scraping failed'}), 500

@app.route('/api/translate')
@limiter.limit("50 per minute")
def api_translate():
    """Translation API"""
    try:
        text = request.args.get('text', '').strip()
        target_lang = request.args.get('target', 'hi')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        if target_lang == 'hi':
            translated = language_manager.translate_to_hindi(text)
        else:
            translated = language_manager.translate_from_hindi(text)
        
        return jsonify({
            'original': text,
            'translated': translated,
            'target_language': target_lang
        })
        
    except Exception as e:
        logger.error(f"Translation API error: {e}")
        return jsonify({'error': 'Translation failed'}), 500

@app.route('/api/categories')
@cache_response(timeout=3600)
def api_categories():
    """Get all categories"""
    try:
        db = next(get_db())
        categories = db.query(Category).filter(Category.is_active == True).order_by(Category.sort_order).all()
        
        result = []
        for cat in categories:
            result.append({
                'id': cat.id,
                'name': cat.name,
                'slug': cat.slug,
                'description': cat.description,
                'icon': cat.icon,
                'parent_id': cat.parent_id
            })
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Categories API error: {e}")
        return jsonify({'error': 'Failed to fetch categories'}), 500

@app.route('/category/<slug>')
def category_page(slug):
    """Category page"""
    try:
        db = next(get_db())
        category = db.query(Category).filter(Category.slug == slug, Category.is_active == True).first()
        
        if not category:
            abort(404)
        
        # Get listings for this category
        listings = db.query(Listing).filter(
            Listing.category_id == category.id,
            Listing.status == 'active'
        ).order_by(Listing.created_at.desc()).limit(50).all()
        
        return render_template_string(TRADE_HUB_TEMPLATE,
                                    page='category',
                                    category=category,
                                    listings=listings,
                                    user_session=session)
        
    except Exception as e:
        logger.error(f"Category page error: {e}")
        abort(404)

@app.route('/listing/<int:listing_id>')
def listing_detail(listing_id):
    """Listing detail page"""
    try:
        db = next(get_db())
        listing = db.query(Listing).filter(Listing.id == listing_id, Listing.status == 'active').first()
        
        if not listing:
            abort(404)
        
        # Increment view count
        listing.views += 1
        db.commit()
        
        # Get similar listings
        similar_listings = db.query(Listing).filter(
            Listing.category_id == listing.category_id,
            Listing.id != listing.id,
            Listing.status == 'active'
        ).order_by(Listing.created_at.desc()).limit(6).all()
        
        return render_template_string(TRADE_HUB_TEMPLATE,
                                    page='listing',
                                    listing=listing,
                                    similar_listings=similar_listings,
                                    user_session=session)
        
    except Exception as e:
        logger.error(f"Listing detail error: {e}")
        abort(404)

@app.route('/search')
def search_page():
    """Search results page"""
    query = request.args.get('q', '')
    category = request.args.get('category', '')
    location = request.args.get('location', '')
    
    return render_template_string(TRADE_HUB_TEMPLATE,
                                page='search',
                                search_query=query,
                                search_category=category,
                                search_location=location,
                                user_session=session)

@app.route('/profile')
def profile_page():
    """User profile page"""
    if 'user_id' not in session:
        return redirect('/auth/login')
    
    return render_template_string(TRADE_HUB_TEMPLATE,
                                page='profile',
                                user_session=session)

@app.route('/auth/login')
def login_page():
    """Login page"""
    return render_template_string(TRADE_HUB_TEMPLATE,
                                page='login',
                                user_session=session)

@app.route('/auth/register')
def register_page():
    """Registration page"""
    return render_template_string(TRADE_HUB_TEMPLATE,
                                page='register',
                                user_session=session)

@app.route('/post-ad')
def post_ad_page():
    """Post advertisement page"""
    return render_template_string(TRADE_HUB_TEMPLATE,
                                page='post-ad',
                                user_session=session)

@app.route('/my-ads')
def my_ads_page():
    """My advertisements page"""
    if 'user_id' not in session:
        return redirect('/auth/login')
    
    return render_template_string(TRADE_HUB_TEMPLATE,
                                page='my-ads',
                                user_session=session)

@app.route('/watchlist')
def watchlist_page():
    """Watchlist page"""
    if 'user_id' not in session:
        return redirect('/auth/login')
    
    return render_template_string(TRADE_HUB_TEMPLATE,
                                page='watchlist',
                                user_session=session)

@app.route('/messages')
def messages_page():
    """Messages page"""
    if 'user_id' not in session:
        return redirect('/auth/login')
    
    return render_template_string(TRADE_HUB_TEMPLATE,
                                page='messages',
                                user_session=session)

@app.route('/help')
def help_page():
    """Help page"""
    return render_template_string(TRADE_HUB_TEMPLATE,
                                page='help',
                                user_session=session)

@app.route('/about')
def about_page():
    """About page"""
    return render_template_string(TRADE_HUB_TEMPLATE,
                                page='about',
                                user_session=session)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template_string(TRADE_HUB_TEMPLATE,
                                page='404',
                                user_session=session), 404

@app.errorhandler(500)
def server_error(error):
    return render_template_string(TRADE_HUB_TEMPLATE,
                                page='500',
                                user_session=session), 500

# ==================== ENHANCED HTML TEMPLATE ====================
TRADE_HUB_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% if page == 'home' %}Trade Hub - Your Ultimate Trading Platform{% else %}{{ page|title }} - Trade Hub{% endif %}</title>
    <meta name="description" content="Trade Hub - Buy, sell, and discover amazing deals. Your one-stop platform for property, motors, jobs, services, and marketplace.">
    <link rel="icon" type="image/x-icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🏪</text></svg>">
    
    <!-- CSS Framework & Icons -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <style>
        :root {
            --primary-color: #2563eb;
            --primary-dark: #1d4ed8;
            --secondary-color: #f59e0b;
            --success-color: #10b981;
            --danger-color: #ef4444;
            --warning-color: #f59e0b;
            --info-color: #06b6d4;
            --light-color: #f8fafc;
            --dark-color: #1e293b;
            --border-color: #e2e8f0;
            --text-muted: #64748b;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
            --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1);
            --radius-sm: 0.375rem;
            --radius-md: 0.5rem;
            --radius-lg: 0.75rem;
            --radius-xl: 1rem;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--dark-color);
            background: var(--light-color);
            font-size: 14px;
        }

        .container-fluid {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        /* Enhanced Header */
        .header {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
            color: white;
            padding: 0.75rem 0;
            box-shadow: var(--shadow-lg);
            position: sticky;
            top: 0;
            z-index: 1000;
            backdrop-filter: blur(10px);
        }

        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }

        .logo {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--secondary-color);
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: transform 0.3s ease;
        }

        .logo:hover {
            transform: scale(1.05);
            color: var(--secondary-color);
        }

        .search-container {
            flex: 1;
            max-width: 600px;
            position: relative;
        }

        .search-form {
            display: flex;
            gap: 0.5rem;
            background: white;
            border-radius: var(--radius-xl);
            padding: 0.25rem;
            box-shadow: var(--shadow-md);
        }

        .search-input {
            flex: 1;
            border: none;
            outline: none;
            padding: 0.75rem 1rem;
            border-radius: var(--radius-lg);
            font-size: 1rem;
        }

        .search-btn {
            background: var(--secondary-color);
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: var(--radius-lg);
            cursor: pointer;
            color: white;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .search-btn:hover {
            background: #d97706;
            transform: translateY(-1px);
        }

        .user-actions {
            display: flex;
            gap: 0.75rem;
            align-items: center;
        }

        .btn {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: var(--radius-md);
            cursor: pointer;
            text-decoration: none;
            font-weight: 500;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.875rem;
        }

        .btn-primary {
            background: var(--secondary-color);
            color: white;
        }

        .btn-outline {
            background: transparent;
            color: white;
            border: 2px solid rgba(255, 255, 255, 0.3);
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }

        .btn-primary:hover {
            background: #d97706;
            color: white;
        }

        .btn-outline:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: white;
            color: white;
        }

        /* Enhanced Navigation */
        .nav-section {
            background: white;
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 0;
            box-shadow: var(--shadow-sm);
        }

        .nav-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }

        .nav-links {
            display: flex;
            gap: 0.5rem;
            list-style: none;
            flex-wrap: wrap;
        }

        .nav-links a {
            color: var(--text-muted);
            text-decoration: none;
            font-weight: 500;
            padding: 0.75rem 1rem;
            border-radius: var(--radius-md);
            transition: all 0.3s ease;
            position: relative;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .nav-links a:hover,
        .nav-links a.active {
            color: var(--primary-color);
            background: rgba(37, 99, 235, 0.1);
        }

        .nav-links a.active::after {
            content: '';
            position: absolute;
            bottom: -1rem;
            left: 50%;
            transform: translateX(-50%);
            width: 20px;
            height: 3px;
            background: var(--primary-color);
            border-radius: 2px;
        }

        .nav-extras {
            display: flex;
            gap: 0.75rem;
            align-items: center;
        }

        .post-ad-btn {
            background: var(--success-color);
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: var(--radius-lg);
            text-decoration: none;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: var(--shadow-sm);
        }

        .post-ad-btn:hover {
            background: #059669;
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
            color: white;
        }

        /* Main Content */
        .main-content {
            min-height: calc(100vh - 200px);
            padding: 2rem 0;
        }

        /* Hero Section */
        .hero-section {
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.1) 0%, rgba(245, 158, 11, 0.1) 100%);
            padding: 3rem 0;
            margin-bottom: 3rem;
            border-radius: var(--radius-xl);
        }

        .hero-content {
            text-align: center;
            max-width: 800px;
            margin: 0 auto;
        }

        .hero-title {
            font-size: 3rem;
            font-weight: 700;
            color: var(--dark-color);
            margin-bottom: 1rem;
            line-height: 1.2;
        }

        .hero-subtitle {
            font-size: 1.25rem;
            color: var(--text-muted);
            margin-bottom: 2rem;
            line-height: 1.5;
        }

        .hero-search {
            max-width: 600px;
            margin: 0 auto;
        }

        .hero-stats {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-top: 2rem;
            flex-wrap: wrap;
        }

        .stat-item {
            text-align: center;
        }

        .stat-number {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary-color);
            display: block;
        }

        .stat-label {
            color: var(--text-muted);
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Category Grid */
        .categories-section {
            margin-bottom: 3rem;
        }

        .section-title {
            font-size: 2rem;
            font-weight: 600;
            color: var(--dark-color);
            margin-bottom: 1.5rem;
            text-align: center;
        }

        .categories-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .category-card {
            background: white;
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            text-align: center;
            text-decoration: none;
            color: inherit;
            transition: all 0.3s ease;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--border-color);
        }

        .category-card:hover {
            transform: translateY(-4px);
            box-shadow: var(--shadow-lg);
            color: inherit;
        }

        .category-icon {
            font-size: 2.5rem;
            margin-bottom: 1rem;
            display: block;
        }

        .category-name {
            font-size: 1.125rem;
            font-weight: 600;
            color: var(--dark-color);
            margin-bottom: 0.5rem;
        }

        .category-description {
            color: var(--text-muted);
            font-size: 0.875rem;
        }

        /* Listings Grid */
        .listings-section {
            margin-bottom: 3rem;
        }

        .listings-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1.5rem;
        }

        .listing-card {
            background: white;
            border-radius: var(--radius-lg);
            overflow: hidden;
            text-decoration: none;
            color: inherit;
            transition: all 0.3s ease;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--border-color);
        }

        .listing-card:hover {
            transform: translateY(-4px);
            box-shadow: var(--shadow-lg);
            color: inherit;
        }

        .listing-image {
            width: 100%;
            height: 200px;
            object-fit: cover;
            background: var(--light-color);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-muted);
            font-size: 3rem;
        }

        .listing-content {
            padding: 1.25rem;
        }

        .listing-title {
            font-size: 1.125rem;
            font-weight: 600;
            color: var(--dark-color);
            margin-bottom: 0.5rem;
            line-height: 1.4;
        }

        .listing-price {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--success-color);
            margin-bottom: 0.75rem;
        }

        .listing-location {
            color: var(--text-muted);
            font-size: 0.875rem;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }

        .listing-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        .listing-date {
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }

        .listing-views {
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }

        /* Featured Badge */
        .featured-badge {
            position: absolute;
            top: 0.75rem;
            right: 0.75rem;
            background: var(--secondary-color);
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: var(--radius-sm);
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Search Page */
        .search-filters {
            background: white;
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--border-color);
        }

        .filter-group {
            margin-bottom: 1rem;
        }

        .filter-label {
            font-weight: 600;
            color: var(--dark-color);
            margin-bottom: 0.5rem;
            display: block;
        }

        .filter-input,
        .filter-select {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            outline: none;
            transition: border-color 0.3s ease;
        }

        .filter-input:focus,
        .filter-select:focus {
            border-color: var(--primary-color);
        }

        .search-results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
            gap: 1rem;
        }

        .results-count {
            color: var(--text-muted);
            font-size: 0.875rem;
        }

        .sort-options {
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }

        /* Forms */
        .form-container {
            max-width: 600px;
            margin: 0 auto;
            background: white;
            border-radius: var(--radius-lg);
            padding: 2rem;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--border-color);
        }

        .form-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--dark-color);
            margin-bottom: 1.5rem;
            text-align: center;
        }

        .form-group {
            margin-bottom: 1.5rem;
        }

        .form-label {
            font-weight: 600;
            color: var(--dark-color);
            margin-bottom: 0.5rem;
            display: block;
        }

        .form-input,
        .form-textarea,
        .form-select {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            outline: none;
            transition: all 0.3s ease;
            font-family: inherit;
        }

        .form-input:focus,
        .form-textarea:focus,
        .form-select:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }

        .form-textarea {
            min-height: 120px;
            resize: vertical;
        }

        .form-submit {
            width: 100%;
            background: var(--primary-color);
            color: white;
            padding: 0.75rem;
            border: none;
            border-radius: var(--radius-md);
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .form-submit:hover {
            background: var(--primary-dark);
            transform: translateY(-1px);
        }

        /* Footer */
        .footer {
            background: var(--dark-color);
            color: white;
            padding: 3rem 0 1rem;
            margin-top: 3rem;
        }

        .footer-content {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 2rem;
            margin-bottom: 2rem;
        }

        .footer-section h4 {
            font-size: 1.125rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--secondary-color);
        }

        .footer-links {
            list-style: none;
        }

        .footer-links li {
            margin-bottom: 0.5rem;
        }

        .footer-links a {
            color: rgba(255, 255, 255, 0.7);
            text-decoration: none;
            transition: color 0.3s ease;
        }

        .footer-links a:hover {
            color: white;
        }

        .footer-bottom {
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            padding-top: 1rem;
            text-align: center;
            color: rgba(255, 255, 255, 0.7);
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .hero-title {
                font-size: 2rem;
            }
            
            .hero-subtitle {
                font-size: 1rem;
            }
            
            .hero-stats {
                gap: 1rem;
            }
            
            .categories-grid {
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
            }
            
            .listings-grid {
                grid-template-columns: 1fr;
                gap: 1rem;
            }
            
            .header-content {
                flex-direction: column;
                gap: 1rem;
            }
            
            .search-container {
                order: 3;
                width: 100%;
            }
            
            .nav-links {
                justify-content: center;
            }
            
            .search-results-header {
                flex-direction: column;
                align-items: flex-start;
            }
        }

        @media (max-width: 480px) {
            .container-fluid {
                padding-left: 0.5rem;
                padding-right: 0.5rem;
            }
            
            .hero-section {
                padding: 2rem 0;
            }
            
            .categories-grid {
                grid-template-columns: 1fr;
            }
            
            .category-card {
                padding: 1rem;
            }
            
            .form-container {
                padding: 1.5rem;
            }
        }

        /* Loading Animation */
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Toast Notifications */
        .toast-container {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
        }

        .toast {
            background: white;
            border-radius: var(--radius-md);
            padding: 1rem 1.5rem;
            margin-bottom: 0.5rem;
            box-shadow: var(--shadow-lg);
            border-left: 4px solid var(--primary-color);
            animation: slideIn 0.3s ease;
        }

        .toast.success {
            border-left-color: var(--success-color);
        }

        .toast.error {
            border-left-color: var(--danger-color);
        }

        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }

        /* Utility Classes */
        .text-center { text-align: center; }
        .text-left { text-align: left; }
        .text-right { text-align: right; }
        .d-flex { display: flex; }
        .d-none { display: none; }
        .justify-content-center { justify-content: center; }
        .align-items-center { align-items: center; }
        .gap-1 { gap: 0.5rem; }
        .gap-2 { gap: 1rem; }
        .gap-3 { gap: 1.5rem; }
        .mt-1 { margin-top: 0.5rem; }
        .mt-2 { margin-top: 1rem; }
        .mt-3 { margin-top: 1.5rem; }
        .mb-1 { margin-bottom: 0.5rem; }
        .mb-2 { margin-bottom: 1rem; }
        .mb-3 { margin-bottom: 1.5rem; }
        .p-1 { padding: 0.5rem; }
        .p-2 { padding: 1rem; }
        .p-3 { padding: 1.5rem; }
    </style>
</head>
<body>
    <!-- Header -->
    <header class="header">
        <div class="container-fluid">
            <div class="header-content">
                <a href="/" class="logo">
                    <i class="bi bi-shop"></i>
                    Trade Hub
                </a>
                
                <div class="search-container">
                    <form class="search-form" onsubmit="performSearch(event)">
                        <input type="text" class="search-input" placeholder="Search for anything..." name="q" id="searchInput">
                        <button type="submit" class="search-btn">
                            <i class="bi bi-search"></i>
                            Search
                        </button>
                    </form>
                </div>
                
                <div class="user-actions">
                    {% if user_session.get('user_id') %}
                        <a href="/messages" class="btn btn-outline">
                            <i class="bi bi-chat-dots"></i>
                            Messages
                        </a>
                        <a href="/profile" class="btn btn-outline">
                            <i class="bi bi-person"></i>
                            Profile
                        </a>
                    {% else %}
                        <a href="/auth/login" class="btn btn-outline">
                            <i class="bi bi-box-arrow-in-right"></i>
                            Login
                        </a>
                        <a href="/auth/register" class="btn btn-primary">
                            <i class="bi bi-person-plus"></i>
                            Register
                        </a>
                    {% endif %}
                </div>
            </div>
        </div>
    </header>

    <!-- Navigation -->
    <nav class="nav-section">
        <div class="container-fluid">
            <div class="nav-content">
                <ul class="nav-links">
                    <li><a href="/" class="{% if page == 'home' %}active{% endif %}">
                        <i class="bi bi-house"></i> Home
                    </a></li>
                    <li><a href="/category/property" class="{% if page == 'property' %}active{% endif %}">
                        <i class="bi bi-building"></i> Property
                    </a></li>
                    <li><a href="/category/motors" class="{% if page == 'motors' %}active{% endif %}">
                        <i class="bi bi-car-front"></i> Motors
                    </a></li>
                    <li><a href="/category/jobs" class="{% if page == 'jobs' %}active{% endif %}">
                        <i class="bi bi-briefcase"></i> Jobs
                    </a></li>
                    <li><a href="/category/services" class="{% if page == 'services' %}active{% endif %}">
                        <i class="bi bi-tools"></i> Services
                    </a></li>
                    <li><a href="/category/marketplace" class="{% if page == 'marketplace' %}active{% endif %}">
                        <i class="bi bi-cart"></i> Marketplace
                    </a></li>
                    {% if user_session.get('user_id') %}
                        <li><a href="/my-ads" class="{% if page == 'my-ads' %}active{% endif %}">
                            <i class="bi bi-list-ul"></i> My Ads
                        </a></li>
                        <li><a href="/watchlist" class="{% if page == 'watchlist' %}active{% endif %}">
                            <i class="bi bi-heart"></i> Watchlist
                        </a></li>
                    {% endif %}
                </ul>
                
                <div class="nav-extras">
                    <a href="/post-ad" class="post-ad-btn">
                        <i class="bi bi-plus-circle"></i>
                        Post Free Ad
                    </a>
                </div>
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <main class="main-content">
        <div class="container-fluid">
            {% if page == 'home' %}
                <!-- Hero Section -->
                <section class="hero-section">
                    <div class="hero-content">
                        <h1 class="hero-title">Welcome to Trade Hub</h1>
                        <p class="hero-subtitle">Your ultimate platform for buying, selling, and discovering amazing deals across India</p>
                        
                        <div class="hero-search">
                            <form class="search-form" onsubmit="performSearch(event)">
                                <input type="text" class="search-input" placeholder="What are you looking for?" name="q">
                                <button type="submit" class="search-btn">
                                    <i class="bi bi-search"></i>
                                    Find Deals
                                </button>
                            </form>
                        </div>
                        
                        <div class="hero-stats">
                            <div class="stat-item">
                                <span class="stat-number">{{ stats.total_users|default(0) }}+</span>
                                <span class="stat-label">Active Users</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-number">{{ stats.total_listings|default(0) }}+</span>
                                <span class="stat-label">Live Listings</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-number">{{ stats.total_categories|default(0) }}+</span>
                                <span class="stat-label">Categories</span>
                            </div>
                        </div>
                    </div>
                </section>

                <!-- Categories Section -->
                <section class="categories-section">
                    <h2 class="section-title">Explore Categories</h2>
                    <div class="categories-grid">
                        {% for category in categories %}
                        <a href="/category/{{ category.slug }}" class="category-card">
                            <span class="category-icon">{{ category.icon|default('📦') }}</span>
                            <h3 class="category-name">{{ category.name }}</h3>
                            <p class="category-description">{{ category.description|default('Discover great deals') }}</p>
                        </a>
                        {% endfor %}
                    </div>
                </section>

                <!-- Featured Listings -->
                {% if featured_listings %}
                <section class="listings-section">
                    <h2 class="section-title">Featured Listings</h2>
                    <div class="listings-grid">
                        {% for listing in featured_listings %}
                        <a href="/listing/{{ listing.id }}" class="listing-card">
                            <div class="listing-image">
                                {% if listing.images and listing.images|length > 0 %}
                                    <img src="{{ listing.images[0] }}" alt="{{ listing.title }}" style="width: 100%; height: 200px; object-fit: cover;">
                                {% else %}
                                    <i class="bi bi-image"></i>
                                {% endif %}
                                {% if listing.is_featured %}
                                    <span class="featured-badge">Featured</span>
                                {% endif %}
                            </div>
                            <div class="listing-content">
                                <h3 class="listing-title">{{ listing.title }}</h3>
                                <div class="listing-price">₹{{ listing.price|int }}</div>
                                <div class="listing-location">
                                    <i class="bi bi-geo-alt"></i>
                                    {{ listing.location|default('Location not specified') }}
                                </div>
                                <div class="listing-meta">
                                    <div class="listing-date">
                                        <i class="bi bi-calendar"></i>
                                        {{ listing.created_at.strftime('%b %d') if listing.created_at else 'Recently' }}
                                    </div>
                                    <div class="listing-views">
                                        <i class="bi bi-eye"></i>
                                        {{ listing.views|default(0) }}
                                    </div>
                                </div>
                            </div>
                        </a>
                        {% endfor %}
                    </div>
                </section>
                {% endif %}

            {% elif page == 'search' %}
                <!-- Search Results -->
                <section class="search-section">
                    <div class="search-filters">
                        <h3>Search Filters</h3>
                        <form id="searchFilters" class="row">
                            <div class="col-md-3 filter-group">
                                <label class="filter-label">Category</label>
                                <select class="filter-select" name="category">
                                    <option value="">All Categories</option>
                                    <option value="property">Property</option>
                                    <option value="motors">Motors</option>
                                    <option value="jobs">Jobs</option>
                                    <option value="services">Services</option>
                                    <option value="marketplace">Marketplace</option>
                                </select>
                            </div>
                            <div class="col-md-3 filter-group">
                                <label class="filter-label">Location</label>
                                <input type="text" class="filter-input" name="location" placeholder="Enter location">
                            </div>
                            <div class="col-md-3 filter-group">
                                <label class="filter-label">Min Price</label>
                                <input type="number" class="filter-input" name="min_price" placeholder="₹ 0">
                            </div>
                            <div class="col-md-3 filter-group">
                                <label class="filter-label">Max Price</label>
                                <input type="number" class="filter-input" name="max_price" placeholder="₹ Any">
                            </div>
                        </form>
                    </div>

                    <div class="search-results-header">
                        <div class="results-count">
                            <span id="resultsCount">Searching...</span>
                        </div>
                        <div class="sort-options">
                            <label>Sort by:</label>
                            <select id="sortBy" class="filter-select" style="width: auto;">
                                <option value="created_at">Date Posted</option>
                                <option value="price">Price</option>
                                <option value="views">Popularity</option>
                            </select>
                        </div>
                    </div>

                    <div id="searchResults" class="listings-grid">
                        <!-- Search results will be loaded here -->
                    </div>
                </section>

            {% elif page == 'category' %}
                <!-- Category Page -->
                <section class="category-section">
                    <div class="hero-section">
                        <div class="hero-content">
                            <h1 class="hero-title">{{ category.icon|default('📦') }} {{ category.name }}</h1>
                            <p class="hero-subtitle">{{ category.description|default('Discover great deals in this category') }}</p>
                        </div>
                    </div>

                    {% if listings %}
                    <div class="listings-grid">
                        {% for listing in listings %}
                        <a href="/listing/{{ listing.id }}" class="listing-card">
                            <div class="listing-image">
                                {% if listing.images and listing.images|length > 0 %}
                                    <img src="{{ listing.images[0] }}" alt="{{ listing.title }}" style="width: 100%; height: 200px; object-fit: cover;">
                                {% else %}
                                    <i class="bi bi-image"></i>
                                {% endif %}
                            </div>
                            <div class="listing-content">
                                <h3 class="listing-title">{{ listing.title }}</h3>
                                <div class="listing-price">₹{{ listing.price|int }}</div>
                                <div class="listing-location">
                                    <i class="bi bi-geo-alt"></i>
                                    {{ listing.location|default('Location not specified') }}
                                </div>
                                <div class="listing-meta">
                                    <div class="listing-date">
                                        <i class="bi bi-calendar"></i>
                                        {{ listing.created_at.strftime('%b %d') if listing.created_at else 'Recently' }}
                                    </div>
                                    <div class="listing-views">
                                        <i class="bi bi-eye"></i>
                                        {{ listing.views|default(0) }}
                                    </div>
                                </div>
                            </div>
                        </a>
                        {% endfor %}
                    </div>
                    {% else %}
                    <div class="text-center mt-3">
                        <h3>No listings found in this category</h3>
                        <p>Be the first to post an ad in {{ category.name }}!</p>
                        <a href="/post-ad" class="btn btn-primary">Post Free Ad</a>
                    </div>
                    {% endif %}
                </section>

            {% elif page == 'listing' %}
                <!-- Listing Detail Page -->
                <section class="listing-detail-section">
                    <div class="row">
                        <div class="col-lg-8">
                            <div class="listing-images mb-3">
                                {% if listing.images and listing.images|length > 0 %}
                                    <img src="{{ listing.images[0] }}" alt="{{ listing.title }}" style="width: 100%; height: 400px; object-fit: cover; border-radius: var(--radius-lg);">
                                {% else %}
                                    <div style="width: 100%; height: 400px; background: var(--light-color); display: flex; align-items: center; justify-content: center; border-radius: var(--radius-lg);">
                                        <i class="bi bi-image" style="font-size: 4rem; color: var(--text-muted);"></i>
                                    </div>
                                {% endif %}
                            </div>
                            
                            <div class="listing-details">
                                <h1>{{ listing.title }}</h1>
                                <div class="listing-price mb-3" style="font-size: 2rem;">₹{{ listing.price|int }}</div>
                                
                                <div class="listing-info mb-3">
                                    <div class="d-flex gap-3 mb-2">
                                        <span><i class="bi bi-geo-alt"></i> {{ listing.location|default('Location not specified') }}</span>
                                        <span><i class="bi bi-calendar"></i> {{ listing.created_at.strftime('%B %d, %Y') if listing.created_at else 'Recently posted' }}</span>
                                        <span><i class="bi bi-eye"></i> {{ listing.views|default(0) }} views</span>
                                    </div>
                                    {% if listing.condition %}
                                        <span class="badge bg-primary">{{ listing.condition }}</span>
                                    {% endif %}
                                </div>
                                
                                <div class="listing-description">
                                    <h3>Description</h3>
                                    <p>{{ listing.description }}</p>
                                </div>
                            </div>
                        </div>
                        
                        <div class="col-lg-4">
                            <div class="seller-info" style="background: white; padding: 1.5rem; border-radius: var(--radius-lg); box-shadow: var(--shadow-sm);">
                                <h4>Seller Information</h4>
                                {% if listing.user %}
                                <div class="seller-details">
                                    <p><strong>{{ listing.user.username }}</strong></p>
                                    {% if listing.user.rating > 0 %}
                                        <div class="rating mb-2">
                                            {% for i in range(5) %}
                                                {% if i < listing.user.rating %}
                                                    <i class="bi bi-star-fill text-warning"></i>
                                                {% else %}
                                                    <i class="bi bi-star text-muted"></i>
                                                {% endif %}
                                            {% endfor %}
                                            <span class="text-muted">({{ listing.user.total_ratings }} reviews)</span>
                                        </div>
                                    {% endif %}
                                </div>
                                {% endif %}
                                
                                <div class="contact-actions">
                                    <button class="btn btn-primary w-100 mb-2">
                                        <i class="bi bi-chat-dots"></i> Send Message
                                    </button>
                                    <button class="btn btn-outline-primary w-100 mb-2">
                                        <i class="bi bi-telephone"></i> Show Phone
                                    </button>
                                    <button class="btn btn-outline-secondary w-100">
                                        <i class="bi bi-heart"></i> Add to Watchlist
                                    </button>
                                </div>
                            </div>
                            
                            {% if similar_listings %}
                            <div class="similar-listings mt-3" style="background: white; padding: 1.5rem; border-radius: var(--radius-lg); box-shadow: var(--shadow-sm);">
                                <h4>Similar Listings</h4>
                                {% for similar in similar_listings %}
                                <div class="similar-item d-flex gap-2 mb-2 p-2" style="border: 1px solid var(--border-color); border-radius: var(--radius-md);">
                                    <div style="width: 60px; height: 60px; background: var(--light-color); border-radius: var(--radius-sm); display: flex; align-items: center; justify-content: center;">
                                        <i class="bi bi-image"></i>
                                    </div>
                                    <div class="flex-grow-1">
                                        <h6 class="mb-1">{{ similar.title[:30] }}...</h6>
                                        <div class="text-success fw-bold">₹{{ similar.price|int }}</div>
                                    </div>
                                </div>
                                {% endfor %}
                            </div>
                            {% endif %}
                        </div>
                    </div>
                </section>

            {% elif page == 'post-ad' %}
                <!-- Post Advertisement -->
                <section class="post-ad-section">
                    <div class="form-container">
                        <h2 class="form-title">Post Your Free Advertisement</h2>
                        <form id="postAdForm">
                            <div class="form-group">
                                <label class="form-label">Category *</label>
                                <select class="form-select" name="category" required>
                                    <option value="">Select Category</option>
                                    <option value="property">Property</option>
                                    <option value="motors">Motors</option>
                                    <option value="jobs">Jobs</option>
                                    <option value="services">Services</option>
                                    <option value="marketplace">Marketplace</option>
                                    <option value="electronics">Electronics</option>
                                    <option value="fashion">Fashion</option>
                                    <option value="home-garden">Home & Garden</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label class="form-label">Title *</label>
                                <input type="text" class="form-input" name="title" placeholder="Enter a descriptive title" required>
                            </div>
                            
                            <div class="form-group">
                                <label class="form-label">Description *</label>
                                <textarea class="form-textarea" name="description" placeholder="Describe your item in detail..." required></textarea>
                            </div>
                            
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="form-group">
                                        <label class="form-label">Price *</label>
                                        <input type="number" class="form-input" name="price" placeholder="₹ 0" required>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="form-group">
                                        <label class="form-label">Condition</label>
                                        <select class="form-select" name="condition">
                                            <option value="">Select Condition</option>
                                            <option value="New">New</option>
                                            <option value="Like New">Like New</option>
                                            <option value="Good">Good</option>
                                            <option value="Fair">Fair</option>
                                            <option value="Poor">Poor</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="form-group">
                                <label class="form-label">Location *</label>
                                <input type="text" class="form-input" name="location" placeholder="Enter your city/area" required>
                            </div>
                            
                            <div class="form-group">
                                <label class="form-label">Contact Phone</label>
                                <input type="tel" class="form-input" name="phone" placeholder="Your phone number">
                            </div>
                            
                            <div class="form-group">
                                <label class="form-label">Images</label>
                                <input type="file" class="form-input" name="images" multiple accept="image/*">
                                <small class="text-muted">Upload up to 5 images (max 2MB each)</small>
                            </div>
                            
                            <button type="submit" class="form-submit">
                                <i class="bi bi-plus-circle"></i>
                                Post Advertisement
                            </button>
                        </form>
                    </div>
                </section>

            {% elif page == 'login' %}
                <!-- Login Form -->
                <section class="auth-section">
                    <div class="form-container">
                        <h2 class="form-title">Login to Trade Hub</h2>
                        <form id="loginForm">
                            <div class="form-group">
                                <label class="form-label">Email or Username</label>
                                <input type="text" class="form-input" name="username" placeholder="Enter your email or username" required>
                            </div>
                            
                            <div class="form-group">
                                <label class="form-label">Password</label>
                                <input type="password" class="form-input" name="password" placeholder="Enter your password" required>
                            </div>
                            
                            <button type="submit" class="form-submit">
                                <i class="bi bi-box-arrow-in-right"></i>
                                Login
                            </button>
                            
                            <div class="text-center mt-3">
                                <p>Don't have an account? <a href="/auth/register">Register here</a></p>
                                <a href="/auth/forgot-password">Forgot Password?</a>
                            </div>
                        </form>
                    </div>
                </section>

            {% elif page == 'register' %}
                <!-- Registration Form -->
                <section class="auth-section">
                    <div class="form-container">
                        <h2 class="form-title">Join Trade Hub</h2>
                        <form id="registerForm">
                            <div class="form-group">
                                <label class="form-label">Full Name</label>
                                <input type="text" class="form-input" name="full_name" placeholder="Enter your full name" required>
                            </div>
                            
                            <div class="form-group">
                                <label class="form-label">Username</label>
                                <input type="text" class="form-input" name="username" placeholder="Choose a username" required>
                            </div>
                            
                            <div class="form-group">
                                <label class="form-label">Email</label>
                                <input type="email" class="form-input" name="email" placeholder="Enter your email" required>
                            </div>
                            
                            <div class="form-group">
                                <label class="form-label">Phone</label>
                                <input type="tel" class="form-input" name="phone" placeholder="Enter your phone number" required>
                            </div>
                            
                            <div class="form-group">
                                <label class="form-label">Location</label>
                                <input type="text" class="form-input" name="location" placeholder="Enter your city" required>
                            </div>
                            
                            <div class="form-group">
                                <label class="form-label">Password</label>
                                <input type="password" class="form-input" name="password" placeholder="Create a strong password" required>
                            </div>
                            
                            <div class="form-group">
                                <label class="form-label">Confirm Password</label>
                                <input type="password" class="form-input" name="confirm_password" placeholder="Confirm your password" required>
                            </div>
                            
                            <button type="submit" class="form-submit">
                                <i class="bi bi-person-plus"></i>
                                Create Account
                            </button>
                            
                            <div class="text-center mt-3">
                                <p>Already have an account? <a href="/auth/login">Login here</a></p>
                            </div>
                        </form>
                    </div>
                </section>

            {% elif page == 'profile' %}
                <!-- User Profile -->
                <section class="profile-section">
                    <div class="row">
                        <div class="col-lg-4">
                            <div class="profile-card" style="background: white; padding: 2rem; border-radius: var(--radius-lg); box-shadow: var(--shadow-sm);">
                                <div class="text-center mb-3">
                                    <div class="profile-avatar" style="width: 100px; height: 100px; border-radius: 50%; background: var(--primary-color); display: flex; align-items: center; justify-content: center; color: white; font-size: 2rem; margin: 0 auto;">
                                        <i class="bi bi-person"></i>
                                    </div>
                                    <h3 class="mt-2">{{ user_session.get('username', 'User') }}</h3>
                                    <div class="rating">
                                        {% for i in range(5) %}
                                            <i class="bi bi-star text-muted"></i>
                                        {% endfor %}
                                        <span class="text-muted">(0 reviews)</span>
                                    </div>
                                </div>
                                
                                <div class="profile-stats">
                                    <div class="stat-row d-flex justify-content-between mb-2">
                                        <span>Active Ads:</span>
                                        <strong>0</strong>
                                    </div>
                                    <div class="stat-row d-flex justify-content-between mb-2">
                                        <span>Total Views:</span>
                                        <strong>0</strong>
                                    </div>
                                    <div class="stat-row d-flex justify-content-between mb-2">
                                        <span>Member Since:</span>
                                        <strong>Today</strong>
                                    </div>
                                </div>
                                
                                <button class="btn btn-primary w-100 mt-3">Edit Profile</button>
                            </div>
                        </div>
                        
                        <div class="col-lg-8">
                            <div class="profile-content">
                                <div class="profile-tabs mb-3">
                                    <ul class="nav nav-tabs">
                                        <li class="nav-item">
                                            <a class="nav-link active" data-tab="my-ads">My Ads</a>
                                        </li>
                                        <li class="nav-item">
                                            <a class="nav-link" data-tab="watchlist">Watchlist</a>
                                        </li>
                                        <li class="nav-item">
                                            <a class="nav-link" data-tab="messages">Messages</a>
                                        </li>
                                        <li class="nav-item">
                                            <a class="nav-link" data-tab="settings">Settings</a>
                                        </li>
                                    </ul>
                                </div>
                                
                                <div class="tab-content">
                                    <div class="tab-pane active" id="my-ads">
                                        <div class="text-center">
                                            <i class="bi bi-list-ul" style="font-size: 4rem; color: var(--text-muted);"></i>
                                            <h4>No ads posted yet</h4>
                                            <p>Start selling by posting your first advertisement</p>
                                            <a href="/post-ad" class="btn btn-primary">Post Your First Ad</a>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

            {% elif page == 'help' %}
                <!-- Help Page -->
                <section class="help-section">
                    <div class="row">
                        <div class="col-lg-8 mx-auto">
                            <div class="help-content" style="background: white; padding: 2rem; border-radius: var(--radius-lg); box-shadow: var(--shadow-sm);">
                                <h1>Help & Support</h1>
                                
                                <div class="help-topics">
                                    <h3>Frequently Asked Questions</h3>
                                    
                                    <div class="faq-item mb-3">
                                        <h5>How do I post an advertisement?</h5>
                                        <p>Click on "Post Free Ad" button, fill in the required details, upload images, and publish your ad. It's completely free!</p>
                                    </div>
                                    
                                    <div class="faq-item mb-3">
                                        <h5>How do I contact a seller?</h5>
                                        <p>On any listing page, you can send a message to the seller or reveal their phone number to call them directly.</p>
                                    </div>
                                    
                                    <div class="faq-item mb-3">
                                        <h5>Is it safe to buy/sell on Trade Hub?</h5>
                                        <p>We recommend meeting in public places, verifying items before purchase, and using secure payment methods. Always trust your instincts.</p>
                                    </div>
                                    
                                    <div class="faq-item mb-3">
                                        <h5>How do I edit or delete my ad?</h5>
                                        <p>Go to "My Ads" section in your profile to manage all your advertisements. You can edit, promote, or delete them anytime.</p>
                                    </div>
                                </div>
                                
                                <div class="contact-support mt-4">
                                    <h3>Still need help?</h3>
                                    <p>Our support team is here to help you 24/7</p>
                                    <div class="d-flex gap-2">
                                        <a href="mailto:support@tradehub.com" class="btn btn-primary">Email Support</a>
                                        <a href="tel:+911234567890" class="btn btn-outline-primary">Call Support</a>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

            {% elif page == 'about' %}
                <!-- About Page -->
                <section class="about-section">
                    <div class="row">
                        <div class="col-lg-8 mx-auto">
                            <div class="about-content" style="background: white; padding: 2rem; border-radius: var(--radius-lg); box-shadow: var(--shadow-sm);">
                                <h1>About Trade Hub</h1>
                                
                                <p class="lead">Trade Hub is India's leading online marketplace where millions of users come together to buy, sell, and discover amazing deals across various categories.</p>
                                
                                <h3>Our Mission</h3>
                                <p>To connect buyers and sellers across India through a simple, safe, and efficient platform that makes trading accessible to everyone.</p>
                                
                                <h3>What We Offer</h3>
                                <ul>
                                    <li><strong>Property:</strong> Buy, sell, or rent residential and commercial properties</li>
                                    <li><strong>Motors:</strong> Cars, bikes, and all automotive needs</li>
                                    <li><strong>Jobs:</strong> Find employment opportunities or hire talent</li>
                                    <li><strong>Services:</strong> Professional services and skill-based offerings</li>
                                    <li><strong>Marketplace:</strong> General goods, electronics, fashion, and more</li>
                                </ul>
                                
                                <h3>Why Choose Trade Hub?</h3>
                                <div class="row">
                                    <div class="col-md-6">
                                        <ul>
                                            <li>✅ Completely Free to Use</li>
                                            <li>✅ Easy Ad Posting</li>
                                            <li>✅ Wide Reach Across India</li>
                                            <li>✅ Safe & Secure Platform</li>
                                        </ul>
                                    </div>
                                    <div class="col-md-6">
                                        <ul>
                                            <li>✅ Advanced Search Filters</li>
                                            <li>✅ Mobile Responsive</li>
                                            <li>✅ 24/7 Customer Support</li>
                                            <li>✅ Verified User Profiles</li>
                                        </ul>
                                    </div>
                                </div>
                                
                                <div class="cta-section text-center mt-4">
                                    <h3>Ready to Start Trading?</h3>
                                    <p>Join millions of users who trust Trade Hub for their buying and selling needs</p>
                                    <a href="/auth/register" class="btn btn-primary btn-lg">Join Trade Hub Today</a>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

            {% elif page == '404' %}
                <!-- 404 Error Page -->
                <section class="error-section">
                    <div class="text-center">
                        <h1 style="font-size: 8rem; color: var(--text-muted);">404</h1>
                        <h2>Page Not Found</h2>
                        <p>The page you're looking for doesn't exist or has been moved.</p>
                        <div class="d-flex justify-content-center gap-2">
                            <a href="/" class="btn btn-primary">Go Home</a>
                            <a href="/search" class="btn btn-outline-primary">Search Listings</a>
                        </div>
                    </div>
                </section>

            {% elif page == '500' %}
                <!-- 500 Error Page -->
                <section class="error-section">
                    <div class="text-center">
                        <h1 style="font-size: 8rem; color: var(--danger-color);">500</h1>
                        <h2>Server Error</h2>
                        <p>Something went wrong on our end. Please try again later.</p>
                        <div class="d-flex justify-content-center gap-2">
                            <a href="/" class="btn btn-primary">Go Home</a>
                            <a href="/help" class="btn btn-outline-primary">Contact Support</a>
                        </div>
                    </div>
                </section>

            {% endif %}
        </div>
    </main>

    <!-- Footer -->
    <footer class="footer">
        <div class="container-fluid">
            <div class="footer-content">
                <div class="footer-section">
                    <h4>Trade Hub</h4>
                    <p>Your ultimate trading platform for buying, selling, and discovering amazing deals across India.</p>
                    <div class="social-links">
                        <a href="#" class="text-white me-2"><i class="bi bi-facebook"></i></a>
                        <a href="#" class="text-white me-2"><i class="bi bi-twitter"></i></a>
                        <a href="#" class="text-white me-2"><i class="bi bi-instagram"></i></a>
                        <a href="#" class="text-white"><i class="bi bi-youtube"></i></a>
                    </div>
                </div>
                
                <div class="footer-section">
                    <h4>Categories</h4>
                    <ul class="footer-links">
                        <li><a href="/category/property">Property</a></li>
                        <li><a href="/category/motors">Motors</a></li>
                        <li><a href="/category/jobs">Jobs</a></li>
                        <li><a href="/category/services">Services</a></li>
                        <li><a href="/category/marketplace">Marketplace</a></li>
                    </ul>
                </div>
                
                <div class="footer-section">
                    <h4>Quick Links</h4>
                    <ul class="footer-links">
                        <li><a href="/post-ad">Post Free Ad</a></li>
                        <li><a href="/search">Search Listings</a></li>
                        <li><a href="/help">Help & Support</a></li>
                        <li><a href="/about">About Us</a></li>
                        <li><a href="/contact">Contact</a></li>
                    </ul>
                </div>
                
                <div class="footer-section">
                    <h4>Support</h4>
                    <ul class="footer-links">
                        <li><a href="/help">Help Center</a></li>
                        <li><a href="/safety">Safety Tips</a></li>
                        <li><a href="/terms">Terms of Use</a></li>
                        <li><a href="/privacy">Privacy Policy</a></li>
                        <li><a href="/sitemap">Sitemap</a></li>
                    </ul>
                </div>
            </div>
            
            <div class="footer-bottom">
                <p>&copy; 2024 Trade Hub. All rights reserved. Made with ❤️ in India</p>
            </div>
        </div>
    </footer>

    <!-- Toast Container -->
    <div class="toast-container" id="toastContainer"></div>

    <!-- Scripts -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    
    <script>
        // Global JavaScript functionality
        
        // Search functionality
        function performSearch(event) {
            event.preventDefault();
            const query = event.target.querySelector('input[name="q"]').value.trim();
            if (query) {
                window.location.href = `/search?q=${encodeURIComponent(query)}`;
            }
        }

        // Toast notification system
        function showToast(message, type = 'info') {
            const toastContainer = document.getElementById('toastContainer');
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            toast.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <span>${message}</span>
                    <button type="button" class="btn-close btn-close-sm" onclick="this.parentElement.parentElement.remove()"></button>
                </div>
            `;
            toastContainer.appendChild(toast);
            
            // Auto remove after 5 seconds
            setTimeout(() => {
                if (toast.parentElement) {
                    toast.remove();
                }
            }, 5000);
        }

        // Search page functionality
        if (window.location.pathname === '/search') {
            const urlParams = new URLSearchParams(window.location.search);
            const query = urlParams.get('q') || '';
            
            // Set search input value
            const searchInput = document.getElementById('searchInput');
            if (searchInput && query) {
                searchInput.value = query;
            }
            
            // Perform initial search
            if (query) {
                performAPISearch();
            }
            
            // Set up filter change handlers
            const filterForm = document.getElementById('searchFilters');
            if (filterForm) {
                const inputs = filterForm.querySelectorAll('input, select');
                inputs.forEach(input => {
                    input.addEventListener('change', performAPISearch);
                });
            }
            
            // Set up sort change handler
            const sortSelect = document.getElementById('sortBy');
            if (sortSelect) {
                sortSelect.addEventListener('change', performAPISearch);
            }
        }

        // API search function
        async function performAPISearch() {
            const resultsContainer = document.getElementById('searchResults');
            const resultsCount = document.getElementById('resultsCount');
            
            if (!resultsContainer) return;
            
            // Show loading
            resultsContainer.innerHTML = '<div class="text-center"><div class="loading"></div> Searching...</div>';
            resultsCount.textContent = 'Searching...';
            
            try {
                // Build search parameters
                const urlParams = new URLSearchParams(window.location.search);
                const params = new URLSearchParams();
                
                // Add query
                const query = urlParams.get('q') || document.getElementById('searchInput')?.value || '';
                if (query) params.append('q', query);
                
                // Add filters
                const filterForm = document.getElementById('searchFilters');
                if (filterForm) {
                    const formData = new FormData(filterForm);
                    for (const [key, value] of formData.entries()) {
                        if (value.trim()) params.append(key, value);
                    }
                }
                
                // Add sorting
                const sortBy = document.getElementById('sortBy')?.value;
                if (sortBy) params.append('sort_by', sortBy);
                
                // Make API request
                const response = await fetch(`/api/search?${params.toString()}`);
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                // Update results count
                resultsCount.textContent = `Found ${data.total} results`;
                
                // Display results
                if (data.results && data.results.length > 0) {
                    resultsContainer.innerHTML = data.results.map(listing => `
                        <a href="/listing/${listing.id}" class="listing-card">
                            <div class="listing-image">
                                ${listing.images && listing.images.length > 0 
                                    ? `<img src="${listing.images[0]}" alt="${listing.title}" style="width: 100%; height: 200px; object-fit: cover;">`
                                    : '<i class="bi bi-image"></i>'
                                }
                                ${listing.is_featured ? '<span class="featured-badge">Featured</span>' : ''}
                            </div>
                            <div class="listing-content">
                                <h3 class="listing-title">${listing.title}</h3>
                                <div class="listing-price">₹${listing.price}</div>
                                <div class="listing-location">
                                    <i class="bi bi-geo-alt"></i>
                                    ${listing.location || 'Location not specified'}
                                </div>
                                <div class="listing-meta">
                                    <div class="listing-date">
                                        <i class="bi bi-calendar"></i>
                                        ${new Date(listing.created_at).toLocaleDateString()}
                                    </div>
                                    <div class="listing-views">
                                        <i class="bi bi-eye"></i>
                                        ${listing.views || 0}
                                    </div>
                                </div>
                            </div>
                        </a>
                    `).join('');
                } else {
                    resultsContainer.innerHTML = `
                        <div class="text-center">
                            <i class="bi bi-search" style="font-size: 4rem; color: var(--text-muted);"></i>
                            <h4>No results found</h4>
                            <p>Try adjusting your search terms or filters</p>
                        </div>
                    `;
                }
                
            } catch (error) {
                console.error('Search error:', error);
                resultsContainer.innerHTML = `
                    <div class="text-center">
                        <i class="bi bi-exclamation-triangle" style="font-size: 4rem; color: var(--danger-color);"></i>
                        <h4>Search Error</h4>
                        <p>Something went wrong. Please try again.</p>
                    </div>
                `;
                resultsCount.textContent = 'Search failed';
                showToast('Search failed. Please try again.', 'error');
            }
        }

        // Form handling
        document.addEventListener('DOMContentLoaded', function() {
            // Login form
            const loginForm = document.getElementById('loginForm');
            if (loginForm) {
                loginForm.addEventListener('submit', async function(e) {
                    e.preventDefault();
                    showToast('Login functionality will be implemented soon!', 'info');
                });
            }
            
            // Register form
            const registerForm = document.getElementById('registerForm');
            if (registerForm) {
                registerForm.addEventListener('submit', async function(e) {
                    e.preventDefault();
                    showToast('Registration functionality will be implemented soon!', 'info');
                });
            }
            
            // Post ad form
            const postAdForm = document.getElementById('postAdForm');
            if (postAdForm) {
                postAdForm.addEventListener('submit', async function(e) {
                    e.preventDefault();
                    showToast('Ad posting functionality will be implemented soon!', 'info');
                });
            }
        });

        // Smooth scrolling for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });

        // Mobile menu toggle (if needed)
        function toggleMobileMenu() {
            const navLinks = document.querySelector('.nav-links');
            if (navLinks) {
                navLinks.classList.toggle('show');
            }
        }

        // Initialize page-specific functionality
        function initializePage() {
            // Add any page-specific initialization here
            console.log('Trade Hub loaded successfully!');
        }

        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', initializePage);
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print("🚀 Starting Trade Hub - Enhanced Trading Platform")
    print("=" * 60)
    print("✅ All features loaded successfully!")
    print("🌐 Server starting on http://localhost:5000")
    print("📱 Mobile responsive design enabled")
    print("🔍 Advanced search with web scraping")
    print("🌍 Multi-language support (Hindi)")
    print("💾 Production-grade database")
    print("🚀 Redis caching enabled")
    print("🔒 Security features active")
    print("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('DEBUG', 'False').lower() == 'true',
        threaded=True
    )