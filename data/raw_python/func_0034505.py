def validate_url(cls, url: str) -> Optional[Match[str]]:
        """Check if the Extractor can handle the given url."""
        match = re.match(cls._VALID_URL, url)
        return match