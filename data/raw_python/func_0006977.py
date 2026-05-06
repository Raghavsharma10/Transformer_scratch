def _increment_numeric_suffix(s):
        """Increment (or add) numeric suffix to identifier."""
        if re.match(r".*\d+$", s):
            return re.sub(r"\d+$", lambda n: str(int(n.group(0)) + 1), s)
        return s + "_2"