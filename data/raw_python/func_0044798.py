def _write_zip(self, func_src, fpath):
        """
        Write the function source to a zip file, suitable for upload to
        Lambda.

        Note there's a bit of undocumented magic going on here; Lambda needs
        the execute bit set on the module with the handler in it (i.e. 0755
        or 0555 permissions). There doesn't seem to be *any* documentation on
        how to do this in the Python docs. The only real hint comes from the
        source code of ``zipfile.ZipInfo.from_file()``, which includes:

            st = os.stat(filename)
            ...
            zinfo.external_attr = (st.st_mode & 0xFFFF) << 16  # Unix attributes

        :param func_src: lambda function source
        :type func_src: str
        :param fpath: path to write the zip file at
        :type fpath: str
        """
        # get timestamp for file
        now = datetime.now()
        zi_tup = (now.year, now.month, now.day, now.hour, now.minute,
                  now.second)
        logger.debug('setting zipinfo date to: %s', zi_tup)
        # create a ZipInfo so we can set file attributes/mode
        zinfo = zipfile.ZipInfo('webhook2lambda2sqs_func.py', zi_tup)
        # set file mode
        zinfo.external_attr = 0x0755 << 16
        logger.debug('setting zipinfo file mode to: %s', zinfo.external_attr)
        logger.debug('writing zip file at: %s', fpath)
        with zipfile.ZipFile(fpath, 'w') as z:
            z.writestr(zinfo, func_src)