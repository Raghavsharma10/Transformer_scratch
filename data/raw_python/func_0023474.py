def _upload_resumable(self, upload_info, check_result):
        """Resumable upload and return quickkey

        upload_info -- UploadInfo object
        check_result -- dict of upload/check call result
        """

        resumable_upload = check_result['resumable_upload']

        unit_size = int(resumable_upload['unit_size'])
        number_of_units = int(resumable_upload['number_of_units'])

        # make sure we have calculated the right thing
        logger.debug("number_of_units=%s (expected %s)",
                     number_of_units, len(upload_info.hash_info.units))
        assert len(upload_info.hash_info.units) == number_of_units

        logger.debug("Preparing %d units * %d bytes",
                     number_of_units, unit_size)

        upload_key = None
        retries = UPLOAD_RETRY_COUNT

        all_units_ready = resumable_upload['all_units_ready'] == 'yes'
        bitmap = resumable_upload['bitmap']

        while not all_units_ready and retries > 0:
            upload_key = self._upload_resumable_all(upload_info, bitmap,
                                                    number_of_units, unit_size)

            check_result = self._upload_check(upload_info, resumable=True)

            resumable_upload = check_result['resumable_upload']
            all_units_ready = resumable_upload['all_units_ready'] == 'yes'
            bitmap = resumable_upload['bitmap']

            if not all_units_ready:
                retries -= 1
                logger.debug("Some units failed to upload (%d retries left)",
                             retries)

        if not all_units_ready:
            # Most likely non-retriable
            raise UploadError("Could not upload all units")

        logger.debug("Upload complete, polling for status")

        return self._poll_upload(upload_key, 'upload/resumable')