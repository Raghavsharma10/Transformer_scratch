def _get_album_or_image(json, imgur):
    """Return a gallery image/album depending on what the json represent."""
    if json['is_album']:
        return Gallery_album(json, imgur, has_fetched=False)
    return Gallery_image(json, imgur)