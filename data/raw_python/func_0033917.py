def _make_regex(self):
    """
    Build a re object based on keys in the current dictionary
    """
    return re.compile("|".join(map(re.escape, self.keys())))