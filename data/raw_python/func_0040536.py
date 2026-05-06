def from_credentials(cls: Type[SigningKeyType], salt: Union[str, bytes], password: Union[str, bytes],
                         scrypt_params: Optional[ScryptParams] = None) -> SigningKeyType:
        """
        Create a SigningKey object from credentials

        :param salt: Secret salt passphrase credential
        :param password: Secret password credential
        :param scrypt_params: ScryptParams instance
        """
        if scrypt_params is None:
            scrypt_params = ScryptParams()

        salt = ensure_bytes(salt)
        password = ensure_bytes(password)
        seed = scrypt(password, salt, scrypt_params.N, scrypt_params.r, scrypt_params.p, scrypt_params.seed_length)

        return cls(seed)