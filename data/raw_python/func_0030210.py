def dict(self):
        """A dict that holds key/values for all of the properties in the
        object.

        :return:

        """

        d = {p.key: getattr(self, p.key) for p in self.__mapper__.attrs if p.key not in ('data')}

        d['secret'] = 'not available'

        if self.secret_password:
            try:
                d['secret'] = self.decrypt_secret()
            except AccountDecryptionError:
                pass

        if self.data:
            for k, v in self.data.items():
                d[k] = v

        return {k: v for k, v in d.items()}