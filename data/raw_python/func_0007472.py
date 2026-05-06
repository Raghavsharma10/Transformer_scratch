def generate(self):
        """
        :return: A new token
        :rtype: str
        """
        random_data = os.urandom(100)

        hash_gen = hashlib.new("sha512")
        hash_gen.update(random_data)

        return hash_gen.hexdigest()[:self.token_length]