def sign(self, entry, signer=None):
        """Adds and sign an entry"""
        if (self.get(entry) is not None):
            return
        if (entry.rrsig is None) and (self.private is not None):
            entry.rrsig = DNSSignatureS(entry.name,
                    _TYPE_RRSIG, _CLASS_IN, entry, self.private, signer)
        self.add(entry)
        if (self.private is not None):
            self.add(entry.rrsig)