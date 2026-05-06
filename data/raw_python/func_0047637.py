def md5_checker(self, md5sum, local_file=None, file_object=None):
        """Return True if the local file and the provided `md5sum` are equal.

        If the processed file and the provided md5sum do not match an exception
        is raised indicating the failure.

        :param md5sum: ``str``
        :param local_file: ``str``
        :param file_object: ``BytesIO``
        :return: ``bol``
        """
        def calc_hash():
            """Read the hash.

            :return data_hash.read():
            """
            return file_object.read(128 * md5.block_size)

        if (local_file and os.path.isfile(local_file)) is True or file_object:
            md5 = hashlib.md5()

            if not file_object:
                file_object = open(local_file, 'rb')

            for chk in iter(calc_hash, b''):
                if isinstance(chk, bytes):
                    md5.update(chk)
                else:
                    md5.update(chk.encode('utf-8'))
            else:
                if not file_object:
                    file_object.close()

            lmd5sum = md5.hexdigest()
            msg = 'Hash comparison'
            try:
                if md5sum != lmd5sum:
                    msg = (
                        '%s - CheckSumm Mis-Match "%s" != "%s" for [ %s ]' % (
                            msg, md5sum, lmd5sum, local_file
                        )
                    )
                    raise cloudlib.MD5CheckMismatch(msg)
                else:
                    msg = '%s - CheckSumm verified for [ %s ]' % (
                        msg, local_file
                    )
                    return True
            finally:
                self.log.debug(msg)