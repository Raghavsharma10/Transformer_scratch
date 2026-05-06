def _error_on_missing_application(self, params):
        """Raise an ApplicationNotFoundError if the app is not accessible

        In this case, checks for the java runtime and the RDP jar file.
        """
        if not (os.path.exists('java') or which('java')):
            raise ApplicationNotFoundError(
                "Cannot find java runtime. Is it installed? Is it in your "
                "path?")
        jar_fp = self._get_jar_fp()
        if jar_fp is None:
            raise ApplicationNotFoundError(
                "JAR file not found in current directory and the RDP_JAR_PATH "
                "environment variable is not set.  Please set RDP_JAR_PATH to "
                "the full pathname of the JAR file.")
        if not os.path.exists(jar_fp):
            raise ApplicationNotFoundError(
                "JAR file %s does not exist." % jar_fp)