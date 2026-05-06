def from_id(cls, id):
        """
        This decodes an ID to a public key and verifies the checksum byte. ID
        structure in miniLock is the base58 encoded form of the public key
        appended with a single-byte digest from blake2s of the public key, as a
        simple check-sum.
        """
        decoded = cls.ensure_bytes(base58.b58decode(id))
        assert_type_and_length('id', decoded, bytes, L=33)
        pk = nacl.public.PublicKey(decoded[:-1])
        cs = decoded[-1:]
        if cs != pyblake2.blake2s(pk.encode(), 1).digest():
            raise ValueError("Public Key does not match its attached checksum byte: id='{}', decoded='{}', given checksum='{}', calculated checksum={}".format(id, decoded, cs, pyblake2.blake2s(pk.encode(), 1).digest()))
        return cls(pk)