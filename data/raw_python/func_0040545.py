def from_wif_or_ewif_hex(wif_hex: str, password: Optional[str] = None) -> SigningKeyType:
        """
        Return SigningKey instance from Duniter WIF or EWIF in hexadecimal format

        :param wif_hex: WIF or EWIF string in hexadecimal format
        :param password: Password of EWIF encrypted seed
        """
        wif_bytes = Base58Encoder.decode(wif_hex)

        fi = wif_bytes[0:1]

        if fi == b"\x01":
            return SigningKey.from_wif_hex(wif_hex)
        elif fi == b"\x02" and password is not None:
            return SigningKey.from_ewif_hex(wif_hex, password)
        else:
            raise Exception("Error: Bad format: not WIF nor EWIF")