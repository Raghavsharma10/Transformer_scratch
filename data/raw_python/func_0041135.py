def encode(data: Union[str, bytes]) -> str:
        """
        Return Base58 string from data

        :param data: Bytes or string data
        """
        return ensure_str(base58.b58encode(ensure_bytes(data)))