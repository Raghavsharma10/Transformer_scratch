def hash_host(hostname, salt=None):
        """
        Return a "hashed" form of the hostname, as used by openssh when storing
        hashed hostnames in the known_hosts file.

        @param hostname: the hostname to hash
        @type hostname: str
        @param salt: optional salt to use when hashing (must be 20 bytes long)
        @type salt: str
        @return: the hashed hostname
        @rtype: str
        """
        if salt is None:
            salt = rng.read(SHA.digest_size)
        else:
            if salt.startswith('|1|'):
                salt = salt.split('|')[2]
            salt = base64.decodestring(salt)
        assert len(salt) == SHA.digest_size
        hmac = HMAC.HMAC(salt, hostname, SHA).digest()
        hostkey = '|1|%s|%s' % (base64.encodestring(salt), base64.encodestring(hmac))
        return hostkey.replace('\n', '')