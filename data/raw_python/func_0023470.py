def _upload_instant(self, upload_info, _=None):
        """Instant upload and return quickkey

        Can be used when the file is already stored somewhere in MediaFire

        upload_info -- UploadInfo object
        check_result -- ignored
        """

        result = self._api.upload_instant(
            upload_info.name,
            upload_info.size,
            upload_info.hash_info.file,
            path=upload_info.path,
            folder_key=upload_info.folder_key,
            filedrop_key=upload_info.filedrop_key,
            action_on_duplicate=upload_info.action_on_duplicate
        )

        return UploadResult(
            action='upload/instant',
            quickkey=result['quickkey'],
            filename=result['filename'],
            revision=result['new_device_revision'],
            hash_=upload_info.hash_info.file,
            size=upload_info.size,
            created=None
        )