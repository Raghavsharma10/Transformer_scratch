def _upload_resumable_all(self, upload_info, bitmap,
                              number_of_units, unit_size):
        """Prepare and upload all resumable units and return upload_key

        upload_info -- UploadInfo object
        bitmap -- bitmap node of upload/check
        number_of_units -- number of units requested
        unit_size -- size of a single upload unit in bytes
        """

        fd = upload_info.fd

        upload_key = None

        for unit_id in range(number_of_units):
            upload_status = decode_resumable_upload_bitmap(
                bitmap, number_of_units)

            if upload_status[unit_id]:
                logger.debug("Skipping unit %d/%d - already uploaded",
                             unit_id + 1, number_of_units)
                continue

            logger.debug("Uploading unit %d/%d",
                         unit_id + 1, number_of_units)

            offset = unit_id * unit_size

            with SubsetIO(fd, offset, unit_size) as unit_fd:

                unit_info = _UploadUnitInfo(
                    upload_info=upload_info,
                    hash_=upload_info.hash_info.units[unit_id],
                    fd=unit_fd,
                    uid=unit_id)

                upload_result = self._upload_resumable_unit(unit_info)

                # upload_key is needed for polling
                if upload_key is None:
                    upload_key = upload_result['doupload']['key']

        return upload_key