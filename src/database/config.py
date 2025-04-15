from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import os

# Load variabel dari .env
load_dotenv()

# Ambil DATABASE_URL dari environment
DATABASE_URL = os.getenv("DATABASE_URL")

# Buat engine SQLAlchemy
engine = create_engine(DATABASE_URL)

# Session dan Base ORM
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
