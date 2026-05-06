def extract_emails(results: str, domain: str, fuzzy: bool) -> List[str]:
    """Grab email addresses from raw text data."""
    pattern: Pattern = re.compile(r'([\w.-]+@[\w.-]+)')
    hits: List[str] = pattern.findall(results)
    if fuzzy:
        seed = domain.split('.')[0]
        emails: List[str] = [x.lower() for x in hits if x.split('@')[1].__contains__(seed)]
    else:
        emails: List[str] = [x.lower() for x in hits if x.endswith(domain)]
    return list(set(emails))