def digest(self, alg='sha256', b64=True, strip=True):
        """return a url-safe hash of the string, optionally (and by default) base64-encoded
            alg='sha256'    = the hash algorithm, must be in hashlib
            b64=True        = whether to base64-encode the output
            strip=True      = whether to strip trailing '=' from the base64 output
        Using the default arguments returns a url-safe base64-encoded SHA-256 hash of the string.
        Length of the digest with different algorithms, using b64=True and strip=True:
            * SHA224 = 38 
            * SHA256 = 43 (DEFAULT)
            * SHA384 = 64
            * SHA512 = 86
        """
        import base64, hashlib

        h = hashlib.new(alg)
        h.update(str(self).encode('utf-8'))
        if b64 == True:
            # this returns a string with a predictable amount of = padding at the end
            b = base64.urlsafe_b64encode(h.digest()).decode('ascii')
            if strip == True:
                b = b.rstrip('=')
            return b
        else:
            return h.hexdigest()