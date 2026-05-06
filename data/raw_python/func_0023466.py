def upload(self, fd, name=None, folder_key=None, filedrop_key=None,
               path=None, action_on_duplicate=None):
        """Upload file, returns UploadResult object

        fd -- file-like object to upload from, expects exclusive access
        name -- file name
        folder_key -- folderkey of the target folder
        path -- path to file relative to folder_key
        filedrop_key -- filedrop to use instead of folder_key
        action_on_duplicate -- skip, keep, replace
        """

        # Get file handle content length in the most reliable way
        fd.seek(0, os.SEEK_END)
        size = fd.tell()
        fd.seek(0, os.SEEK_SET)

        if size > UPLOAD_SIMPLE_LIMIT_BYTES:
            resumable = True
        else:
            resumable = False

        logger.debug("Calculating checksum")
        hash_info = compute_hash_info(fd)

        if hash_info.size != size:
            # Has the file changed beween computing the hash
            # and calling upload()?
            raise ValueError("hash_info.size mismatch")

        upload_info = _UploadInfo(fd=fd, name=name, folder_key=folder_key,
                                  hash_info=hash_info, size=size, path=path,
                                  filedrop_key=filedrop_key,
                                  action_on_duplicate=action_on_duplicate)

        # Check whether file is present
        check_result = self._upload_check(upload_info, resumable)

        upload_result = None
        upload_func = None

        folder_key = check_result.get('folder_key', None)
        if folder_key is not None:
            # We know precisely what folder_key to use, drop path
            upload_info.folder_key = folder_key
            upload_info.path = None

        if check_result['hash_exists'] == 'yes':
            # file exists somewhere in MediaFire
            if check_result['in_folder'] == 'yes' and \
                    check_result['file_exists'] == 'yes':
                # file exists in this directory
                different_hash = check_result.get('different_hash', 'no')
                if different_hash == 'no':
                    # file is already there
                    upload_func = self._upload_none

            if not upload_func:
                # different hash or in other folder
                upload_func = self._upload_instant

        if not upload_func:
            if resumable:
                resumable_upload_info = check_result['resumable_upload']
                upload_info.hash_info = compute_hash_info(
                    fd, int(resumable_upload_info['unit_size']))
                upload_func = self._upload_resumable
            else:
                upload_func = self._upload_simple

        # Retry retriable exceptions
        retries = UPLOAD_RETRY_COUNT
        while retries > 0:
            try:
                # Provide check_result to avoid calling API twice
                upload_result = upload_func(upload_info, check_result)
            except (RetriableUploadError, MediaFireConnectionError):
                retries -= 1
                logger.exception("%s failed (%d retries left)",
                                 upload_func.__name__, retries)
                # Refresh check_result for next iteration
                check_result = self._upload_check(upload_info, resumable)
            except Exception:
                logger.exception("%s failed", upload_func)
                break
            else:
                break

        if upload_result is None:
            raise UploadError("Upload failed")

        return upload_result