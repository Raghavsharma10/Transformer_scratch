def generate_slug(value):
    """
    Generates a slug using a Hashid of `value`.

    COPIED from spectator.core.models.SluggedModelMixin() because migrations
    don't make this happen automatically and perhaps the least bad thing is
    to copy the method here, ugh.
    """
    alphabet = app_settings.SLUG_ALPHABET
    salt = app_settings.SLUG_SALT

    hashids = Hashids(alphabet=alphabet, salt=salt, min_length=5)

    return hashids.encode(value)