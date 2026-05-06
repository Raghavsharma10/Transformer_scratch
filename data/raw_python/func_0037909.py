def generate_hash(self):
        """
        Create a unique SHA token/hash using the project SECRET_KEY, URL,
        email address and current datetime.
        """
        from django.utils.hashcompat import sha_constructor
        hash = sha_constructor(settings.SECRET_KEY + self.url + self.email + unicode(datetime.now()) ).hexdigest()
        return hash[::2]