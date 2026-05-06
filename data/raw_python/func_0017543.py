def _make_metadata_request(self, meta_id, metadata_type=None):
        """
        Get the Metadata. The Session initializes with 'COMPACT-DECODED' as the format type. If that returns a DTD error
        then we change to the 'STANDARD-XML' format and try again.
        :param meta_id: The name of the resource, class, or lookup to get metadata for
        :param metadata_type: The RETS metadata type
        :return: list
        """
        # If this metadata _request has already happened, returned the saved result.
        key = '{0!s}:{1!s}'.format(metadata_type, meta_id)
        if key in self.metadata_responses and self.cache_metadata:
            response = self.metadata_responses[key]
        else:
            response = self._request(
                capability='GetMetadata',
                options={
                    'query': {
                        'Type': metadata_type,
                        'ID': meta_id,
                        'Format': self.metadata_format
                    }
                }
            )
            self.metadata_responses[key] = response

        if self.metadata_format == 'COMPACT-DECODED':
            parser = CompactMetadata()
        else:
            parser = StandardXMLetadata()

        try:
            return parser.parse(response=response, metadata_type=metadata_type)
        except RETSException as e:
            # Remove response from cache
            self.metadata_responses.pop(key, None)

            # If the server responds with an invalid parameter for COMPACT-DECODED, try STANDARD-XML
            if self.metadata_format != 'STANDARD-XML' and e.reply_code in ['20513', '20514']:
                self.metadata_responses.pop(key, None)
                self.metadata_format = 'STANDARD-XML'
                return self._make_metadata_request(meta_id=meta_id, metadata_type=metadata_type)
            raise RETSException(e.reply_text, e.reply_code)