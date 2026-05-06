def clean_float(v):
    """Remove commas from a float"""

    if v is None or not str(v).strip():
        return None

    return float(str(v).replace(',', ''))