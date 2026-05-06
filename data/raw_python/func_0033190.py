def _get_jar_fp(self):
        """Returns the full path to the JAR file.

        If the JAR file cannot be found in the current directory and
        the environment variable RDP_JAR_PATH is not set, returns
        None.
        """
        # handles case where the jar file is in the current working directory
        if os.path.exists(self._command):
            return self._command
        # handles the case where the user has specified the location via
        # an environment variable
        elif 'RDP_JAR_PATH' in environ:
            return getenv('RDP_JAR_PATH')
        else:
            return None