def save_apidoc(text: str) -> None:
    """save `text` to apidoc cache"""
    apidoc_local = local.path(APIDOC_LOCAL_FILE)
    if not apidoc_local.dirname.exists():
        apidoc_local.dirname.mkdir()
    with open(apidoc_local, 'w') as f:
        f.write(text)