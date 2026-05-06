def _expand_json(self, j):
        """Decompress the BLOB portion of the usernotes.

        Arguments:
            j: the JSON returned from the wiki page (dict)

        Returns a Dict with the 'blob' key removed and a 'users' key added
        """
        decompressed_json = copy.copy(j)
        decompressed_json.pop('blob', None)  # Remove BLOB portion of JSON

        # Decode and decompress JSON
        compressed_data = base64.b64decode(j['blob'])
        original_json = zlib.decompress(compressed_data).decode('utf-8')

        decompressed_json['users'] = json.loads(original_json)  # Insert users

        return decompressed_json