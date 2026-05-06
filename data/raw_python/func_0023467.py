def _poll_upload(self, upload_key, action):
        """Poll upload until quickkey is found

        upload_key -- upload_key returned by upload/* functions
        """

        if len(upload_key) != UPLOAD_KEY_LENGTH:
            # not a regular 11-char-long upload key
            # There is no API to poll filedrop uploads
            return UploadResult(
                action=action,
                quickkey=None,
                hash_=None,
                filename=None,
                size=None,
                created=None,
                revision=None
            )

        quick_key = None
        while quick_key is None:
            poll_result = self._api.upload_poll(upload_key)
            doupload = poll_result['doupload']

            logger.debug("poll(%s): status=%d, description=%s, filename=%s,"
                         " result=%d",
                         upload_key, int(doupload['status']),
                         doupload['description'], doupload['filename'],
                         int(doupload['result']))

            if int(doupload['result']) != 0:
                break

            if doupload['fileerror'] != '':
                # TODO: we may have to handle this a bit more dramatically
                logger.warning("poll(%s): fileerror=%d", upload_key,
                               int(doupload['fileerror']))
                break

            if int(doupload['status']) == STATUS_NO_MORE_REQUESTS:
                quick_key = doupload['quickkey']
            elif int(doupload['status']) == STATUS_UPLOAD_IN_PROGRESS:
                # BUG: http://forum.mediafiredev.com/showthread.php?588
                raise RetriableUploadError(
                    "Invalid state transition ({})".format(
                        doupload['description']
                    )
                )
            else:
                time.sleep(UPLOAD_POLL_INTERVAL)

        return UploadResult(
            action=action,
            quickkey=doupload['quickkey'],
            hash_=doupload['hash'],
            filename=doupload['filename'],
            size=doupload['size'],
            created=doupload['created'],
            revision=doupload['revision']
        )