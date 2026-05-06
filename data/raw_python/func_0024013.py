def sponsor_tagged_image(sponsor, tag):
    """returns the corresponding url from the tagged image list."""
    if sponsor.files.filter(tag_name=tag).exists():
        return sponsor.files.filter(tag_name=tag).first().tagged_file.item.url
    return ''