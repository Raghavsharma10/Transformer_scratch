def _decrypt(ascii_armor_message: str, signing_key: SigningKey) -> str:
        """
        Decrypt a message from ascii armor format

        :param ascii_armor_message: Utf-8 message
        :param signing_key: SigningKey instance created from credentials
        :return:
        """
        data = signing_key.decrypt_seal(base64.b64decode(ascii_armor_message))

        return data.decode('utf-8')