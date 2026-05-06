def hash(self):
        """Generate a hash value."""
        h = hash_pandas_object(self, index=True)
        return hashlib.md5(h.values.tobytes()).hexdigest()