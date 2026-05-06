def get_upload_key_metadata(self):
        """Generate metadata dictionary from a bucket key."""
        key = self.get_upload_key()
        metadata = key.metadata.copy()

        # Some http header properties which are stored on the key need to be
        # copied to the metadata when updating
        headers = {
            # http header name, key attribute name
            'Cache-Control': 'cache_control',
            'Content-Type': 'content_type',
            'Content-Disposition': 'content_disposition',
            'Content-Encoding': 'content_encoding',
        }

        for header_name, attribute_name in headers.items():
            attribute_value = getattr(key, attribute_name, False)
            if attribute_value:
                metadata.update({b'{0}'.format(header_name):
                                 b'{0}'.format(attribute_value)})
        return metadata