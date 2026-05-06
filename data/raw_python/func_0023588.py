def file_zip(self, keys, confirm_download=None, meta_only=None):
        """file/zip

        http://www.mediafire.com/developers/core_api/1.3/file/#zip
        """
        return self.request('file/zip', QueryParams({
            'keys': keys,
            'confirm_download': confirm_download,
            'meta_only': meta_only
        }))