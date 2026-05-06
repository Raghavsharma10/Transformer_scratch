def _sanitize_instance_name(name, max_length):
    """Instance names must start with a lowercase letter.
    All following characters must be a dash, lowercase letter,
    or digit.
    """
    name = str(name).lower()                # make all letters lowercase
    name = re.sub(r'[^-a-z0-9]', '', name)  # remove invalid characters
    # remove non-lowercase letters from the beginning
    name = re.sub(r'^[^a-z]+', '', name)
    name = name[:max_length]
    name = re.sub(r'-+$', '', name)         # remove hyphens from the end
    return name