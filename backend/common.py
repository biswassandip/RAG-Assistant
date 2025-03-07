from database import SessionLocal, Configuration


# fetches config values from the db
def get_config_value(key):
    """Fetches the latest configuration value from the database."""
    db = SessionLocal()
    config_entry = db.query(Configuration).filter(
        Configuration.key == key).first()
    db.close()
    return config_entry.value if config_entry else None
