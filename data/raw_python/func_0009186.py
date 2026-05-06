def _is_pattern_match(re_pattern, s):
    """Check if a re pattern expression matches an entire string."""
    match = re.match(re_pattern, s, re.I)
    return match.group() == s if match else False