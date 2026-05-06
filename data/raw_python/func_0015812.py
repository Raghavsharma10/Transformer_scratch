def crypter(self):
        """The actual keyczar crypter"""
        if not hasattr(self, '_crypter'):
            # initialise the Keyczar keysets
            if not self.keyset_location:
                raise ValueError('No encrypted keyset location!')
            reader = keyczar.readers.CreateReader(self.keyset_location)
            if self.encrypting_keyset_location:
                encrypting_keyczar = keyczar.Crypter.Read(
                    self.encrypting_keyset_location)
                reader = keyczar.readers.EncryptedReader(reader,
                                                         encrypting_keyczar)
            self._crypter = keyczar.Crypter(reader)
        return self._crypter