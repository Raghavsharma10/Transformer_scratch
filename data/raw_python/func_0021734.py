def decrypt(self, encryption_key, iv, encrypted_data):
        """Decrypt encrypted subtitle data

        @param int subtitle_id
        @param str iv
        @param str encrypted_data
        @return str
        """

        logger.info('Decrypting subtitles with length (%d bytes), key=%r',
            len(encrypted_data), encryption_key)
        return zlib.decompress(aes_decrypt(encryption_key, iv, encrypted_data))