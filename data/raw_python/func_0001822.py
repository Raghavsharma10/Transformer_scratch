def multiple_replace(string, replacements):
    # type: (str, Dict[str,str]) -> str
    """Simultaneously replace multiple strigns in a string

    Args:
        string (str): Input string
        replacements (Dict[str,str]): Replacements dictionary

    Returns:
        str: String with replacements

    """
    pattern = re.compile("|".join([re.escape(k) for k in sorted(replacements, key=len, reverse=True)]), flags=re.DOTALL)
    return pattern.sub(lambda x: replacements[x.group(0)], string)