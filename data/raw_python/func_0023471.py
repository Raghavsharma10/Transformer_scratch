def _upload_simple(self, upload_info, _=None):
        """Simple upload and return quickkey

        Can be used for small files smaller than UPLOAD_SIMPLE_LIMIT_BYTES

        upload_info -- UploadInfo object
        check_result -- ignored
        """

        upload_result = self._api.upload_simple(
            upload_info.fd,
            upload_info.name,
            folder_key=upload_info.folder_key,
            filedrop_key=upload_info.filedrop_key,
            path=upload_info.path,
            file_size=upload_info.size,
            file_hash=upload_info.hash_info.file,
            action_on_duplicate=upload_info.action_on_duplicate)

        logger.debug("upload_result: %s", upload_result)

        upload_key = upload_result['doupload']['key']

        return self._poll_upload(upload_key, 'upload/simple')