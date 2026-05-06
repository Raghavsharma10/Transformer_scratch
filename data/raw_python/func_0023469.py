def _upload_none(self, upload_info, check_result):
        """Dummy upload function for when we don't actually upload"""
        return UploadResult(
            action=None,
            quickkey=check_result['duplicate_quickkey'],
            hash_=upload_info.hash_info.file,
            filename=upload_info.name,
            size=upload_info.size,
            created=None,
            revision=None
        )