def _base_url() -> str:
    """Build base url."""
    _lang: str = "d"
    _type: str = "n"
    _with_suggestions: str = "?"
    return BASE_URI + STBOARD_PATH + _lang + _type + _with_suggestions