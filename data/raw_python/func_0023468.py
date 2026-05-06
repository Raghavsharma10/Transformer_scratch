def _upload_check(self, upload_info, resumable=False):
        """Wrapper around upload/check"""
        return self._api.upload_check(
            filename=upload_info.name,
            size=upload_info.size,
            hash_=upload_info.hash_info.file,
            folder_key=upload_info.folder_key,
            filedrop_key=upload_info.filedrop_key,
            path=upload_info.path,
            resumable=resumable
        )