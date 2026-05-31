import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

# Database URL
# Default to a local postgres DB if not set
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:admin@localhost:5432/phishing_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class ScanResult(Base):
    __tablename__ = "scan_results"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, index=True)
    is_phishing = Column(Boolean)
    confidence = Column(Float)
    whitelisted = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    features = Column(String) # JSON string of features
    security_details = Column(String, nullable=True) # JSON string of SSL/protocol security details
    scan_type = Column(String, default="url") # 'url' or 'email'

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    # First create all tables that don't exist
    Base.metadata.create_all(bind=engine)
    
    # Check if we need to add the scan_type column (simple migration)
    from sqlalchemy import inspect, text
    try:
        inspector = inspect(engine)
        if "scan_results" in inspector.get_table_names():
            columns = [c["name"] for c in inspector.get_columns("scan_results")]
            if "scan_type" not in columns:
                print("Migrating database: adding scan_type column to scan_results...")
                with engine.begin() as conn:
                    conn.execute(text("ALTER TABLE scan_results ADD COLUMN scan_type VARCHAR DEFAULT 'url'"))
                print("Migration successful.")
    except Exception as e:
        print(f"Warning: Database migration failed, recreating tables... Error: {e}")
        try:
            Base.metadata.drop_all(bind=engine)
            Base.metadata.create_all(bind=engine)
        except Exception as recreate_err:
            print(f"Error recreating tables: {recreate_err}")
