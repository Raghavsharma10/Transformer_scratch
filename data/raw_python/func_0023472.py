def _upload_resumable_unit(self, uu_info):
        """Upload a single unit and return raw upload/resumable result

        uu_info -- UploadUnitInfo instance
        """

        # Get actual unit size
        unit_size = uu_info.fd.len

        if uu_info.hash_ is None:
            raise ValueError('UploadUnitInfo.hash_ is now required')

        return self._api.upload_resumable(
            uu_info.fd,
            uu_info.upload_info.size,
            uu_info.upload_info.hash_info.file,
            uu_info.hash_,
            uu_info.uid,
            unit_size,
            filedrop_key=uu_info.upload_info.filedrop_key,
            folder_key=uu_info.upload_info.folder_key,
            path=uu_info.upload_info.path,
            action_on_duplicate=uu_info.upload_info.action_on_duplicate)