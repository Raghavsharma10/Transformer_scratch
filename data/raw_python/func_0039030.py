def hash_data(obj):
    """Generate a SHA1 from a complex object."""
    collect = sha1()
    for text in bytes_iter(obj):
        if isinstance(text, six.text_type):
            text = text.encode('utf-8')
        collect.update(text)
    return collect.hexdigest()