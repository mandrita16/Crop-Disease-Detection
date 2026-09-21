from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./predictions.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    image_path = Column(String, nullable=True)
    disease_prediction = Column(String, nullable=True)
    disease_confidence = Column(Float, nullable=True)
    gradcam_path = Column(String, nullable=True)

    n_val = Column(Float, nullable=True)
    p_val = Column(Float, nullable=True)
    k_val = Column(Float, nullable=True)
    temperature = Column(Float, nullable=True)
    humidity = Column(Float, nullable=True)
    ph = Column(Float, nullable=True)
    rainfall = Column(Float, nullable=True)
    quality_prediction = Column(String, nullable=True)
    quality_confidence = Column(Float, nullable=True)

    fusion_prediction = Column(String, nullable=True)
    fusion_confidence = Column(Float, nullable=True)
    needs_review = Column(Boolean, default=False)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
