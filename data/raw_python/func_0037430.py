def value_from_datadict(self, data, files, name):
        """Returns uploaded file from serialized value."""
        upload = super(StickyUploadWidget, self).value_from_datadict(data, files, name)
        if upload is not None:
            # File was posted or cleared as normal
            return upload
        else:
            # Try the hidden input
            hidden_name = self.get_hidden_name(name)
            value = data.get(hidden_name, None)
            if value is not None:
                upload = open_stored_file(value, self.url)
                if upload is not None:
                    setattr(upload, '_seralized_location', value)
        return upload