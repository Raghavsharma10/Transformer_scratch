def _generate_slug(self, value):
        """
        Generates a slug using a Hashid of `value`.
        """
        alphabet = app_settings.SLUG_ALPHABET
        salt = app_settings.SLUG_SALT

        hashids = Hashids(alphabet=alphabet, salt=salt, min_length=5)

        return hashids.encode(value)