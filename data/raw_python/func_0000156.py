def _compress_json(self, j):
        """Compress the BLOB data portion of the usernotes.

        Arguments:
            j: the JSON in Schema v5 format (dict)

        Returns a dict with the 'users' key removed and 'blob' key added
        """
        compressed_json = copy.copy(j)
        compressed_json.pop('users', None)

        compressed_data = zlib.compress(
            json.dumps(j['users']).encode('utf-8'),
            self.zlib_compression_strength
        )
        b64_data = base64.b64encode(compressed_data).decode('utf-8')

        compressed_json['blob'] = b64_data

        return compressed_json