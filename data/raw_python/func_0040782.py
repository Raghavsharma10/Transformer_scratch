def chmod(cls, path, permission_text):
        """
        :param str permission_text: "ls -l" style permission string. e.g. -rw-r--r--
        """

        try:
            check_file_existence(path)
        except FileNotFoundError:
            _, e, _ = sys.exc_info()  # for python 2.5 compatibility
            logger.debug(e)
            return False

        logger.debug("chmod %s %s" % (path, permission_text))

        os.chmod(path, parseLsPermissionText(permission_text))