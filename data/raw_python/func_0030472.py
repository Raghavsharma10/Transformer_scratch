def update(self, of):
        """Update a file from another file, for copying"""

        # The other values should be set when the file object is created with dataset.bsfile()
        for p in ('mime_type', 'preference', 'state', 'hash', 'modified', 'size', 'contents', 'source_hash', 'data'):
            setattr(self, p, getattr(of, p))

        return self