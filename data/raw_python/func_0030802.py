def clean_int(v):
    """Remove commas from a float"""

    if v is None or not str(v).strip():
        return None

    return int(str(v).replace(',', ''))