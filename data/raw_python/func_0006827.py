def compiled_hash_func(self):
        """Returns compiled hash function based on hash of stringified primary_keys.
        This isn't the most efficient way"""

        def get_primary_key_str(pkey_name):
            return "str(self.{})".format(pkey_name)

        hash_str = "+ ".join([get_primary_key_str(n) for n in self.primary_keys])
        return ALCHEMY_TEMPLATES.hash_function.safe_substitute(concated_primary_key_strs=hash_str)