def pdf_row_limiter(rows, limits=None, **kwargs):
    """
    Limit row passing a value. In this case we dont implementate a best effort
    algorithm because the posibilities are infite with a data text structure
    from a pdf.
    """
    limits = limits or [None, None]

    upper_limit = limits[0] if limits else None
    lower_limit = limits[1] if len(limits) > 1 else None

    return rows[upper_limit: lower_limit]