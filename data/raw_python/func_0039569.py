def put_content(self, content):
        """
        The default content type is set to ``application/octet-stream`` and content encoding set to ``None``.
        """

        self.blob.content_encoding = self.content_encoding
        self.blob.metadata = self.metadata
        return self.blob.upload_from_string(content, content_type=self.content_type)