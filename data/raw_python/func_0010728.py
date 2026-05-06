def normalize(location_name, preserve_commas=False):
    """Normalize *location_name* by stripping punctuation and collapsing
    runs of whitespace, and return the normalized name."""
    def replace(match):
        if preserve_commas and ',' in match.group(0):
            return ','
        return ' '
    return NORMALIZATION_RE.sub(replace, location_name).strip().lower()