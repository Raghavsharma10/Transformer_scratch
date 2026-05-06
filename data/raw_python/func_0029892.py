def get_hashhash(self, username):
        """
        Generate a digest of the htpasswd hash
        """
        return hashlib.sha256(
            self.users.get_hash(username)
        ).hexdigest()