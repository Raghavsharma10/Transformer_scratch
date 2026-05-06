def decrypt_subtitle(self, subtitle):
        """Decrypt encrypted subtitle data in high level model object

        @param crunchyroll.models.Subtitle subtitle
        @return str
        """
        return self.decrypt(self._build_encryption_key(int(subtitle.id)),
            subtitle['iv'][0].text.decode('base64'),
            subtitle['data'][0].text.decode('base64'))