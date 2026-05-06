def get_version(self):
        """
        Returns a tuple representing the installed HAProxy version.

        The value of the tuple is (<major>, <minor>, <patch>), e.g. if HAProxy
        version 1.5.3 is installed, this will return `(1, 5, 3)`.
        """
        command = ["haproxy", "-v"]
        try:
            output = subprocess.check_output(command)
            version_line = output.split("\n")[0]
        except subprocess.CalledProcessError as e:
            logger.error("Could not get HAProxy version: %s", str(e))
            return None

        match = version_re.match(version_line)
        if not match:
            logger.error("Could not parse version from '%s'", version_line)
            return None

        version = (
            int(match.group("major")),
            int(match.group("minor")),
            int(match.group("patch"))
        )

        logger.debug("Got HAProxy version: %s", version)

        return version