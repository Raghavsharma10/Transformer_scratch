def temporary_tag(tag):
    """
    Temporarily tags the repo
    """
    if tag:
        CTX.repo.tag(tag)
    try:
        yield
    finally:
        if tag:
            CTX.repo.remove_tag(tag)