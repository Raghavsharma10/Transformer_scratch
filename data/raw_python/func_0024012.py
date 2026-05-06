def sponsor_image_url(sponsor, name):
    """Returns the corresponding url from the sponsors images"""
    if sponsor.files.filter(name=name).exists():
        # We avoid worrying about multiple matches by always
        # returning the first one.
        return sponsor.files.filter(name=name).first().item.url
    return ''