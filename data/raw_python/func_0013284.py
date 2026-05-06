def hashVariant(cls, gaVariant):
        """
        Produces an MD5 hash of the ga variant object to distinguish
        it from other variants at the same genomic coordinate.
        """
        hash_str = gaVariant.reference_bases + \
            str(tuple(gaVariant.alternate_bases))
        return hashlib.md5(hash_str).hexdigest()