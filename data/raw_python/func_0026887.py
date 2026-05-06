def parse_reading(val: str) -> Optional[float]:
    """ Convert reading value to float (if possible) """
    try:
        return float(val)
    except ValueError:
        logging.warning('Reading of "%s" is not a number', val)
        return None