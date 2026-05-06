def check_url_filetoupload(self):
        # type: () -> None
        """Check if url or file to upload provided for resource and add resource_type and url_type if not supplied

        Returns:
            None
        """
        if self.file_to_upload is None:
            if 'url' in self.data:
                if 'resource_type' not in self.data:
                    self.data['resource_type'] = 'api'
                if 'url_type' not in self.data:
                    self.data['url_type'] = 'api'
            else:
                raise HDXError('Either a url or a file to upload must be supplied!')
        else:
            if 'url' in self.data:
                if self.data['url'] != hdx.data.dataset.Dataset.temporary_url:
                    raise HDXError('Either a url or a file to upload must be supplied not both!')
            if 'resource_type' not in self.data:
                self.data['resource_type'] = 'file.upload'
            if 'url_type' not in self.data:
                self.data['url_type'] = 'upload'
            if 'tracking_summary' in self.data:
                del self.data['tracking_summary']