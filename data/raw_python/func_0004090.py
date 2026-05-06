def compose_nfc(text):
    """Perform unicode composition."""
    if text is None:
        return None
    if not hasattr(compose_nfc, '_tr'):
        compose_nfc._tr = Transliterator.createInstance('Any-NFC')
    return compose_nfc._tr.transliterate(text)