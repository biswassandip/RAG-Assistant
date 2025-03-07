from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from database import SessionLocal, Configuration
from constants import URL_CONFIG_ALL, URL_CONFIG_UPDATE

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get(URL_CONFIG_ALL)
def fetch_all_configs(db: Session = Depends(get_db)):
    """Retrieve all configuration values."""
    configs = db.query(Configuration).all()
    return {config.key: config.value for config in configs}


@router.post(URL_CONFIG_UPDATE)
def update_configs(updated_configs: dict, db: Session = Depends(get_db)):
    """Update multiple configuration values at once."""
    for key, value in updated_configs.items():
        config_entry = db.query(Configuration).filter(
            Configuration.key == key).first()
        if config_entry:
            config_entry.value = value
        else:
            db.add(Configuration(key=key, value=value))
    db.commit()
    return {"message": "Configurations updated successfully!"}
