def generate_slug(value):
    "A copy of spectator.core.models.SluggedModelMixin._generate_slug()"
    alphabet = 'abcdefghijkmnopqrstuvwxyz23456789'
    salt = 'Django Spectator'

    if hasattr(settings, 'SPECTATOR_SLUG_ALPHABET'):
        alphabet = settings.SPECTATOR_SLUG_ALPHABET

    if hasattr(settings, 'SPECTATOR_SLUG_SALT'):
        salt = settings.SPECTATOR_SLUG_SALT

    hashids = Hashids(alphabet=alphabet, salt=salt, min_length=5)

    return hashids.encode(value)