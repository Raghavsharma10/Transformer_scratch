def deconstruct(self):
        """
        Deconstruct the field for Django 1.7+ migrations.
        """
        name, path, args, kwargs = super(BaseEncryptedField, self).deconstruct()
        kwargs.update({
            #'key': self.cipher_key,
            'cipher': self.cipher_name,
            'charset': self.charset,
            'check_armor': self.check_armor,
            'versioned': self.versioned,
        })
        return name, path, args, kwargs