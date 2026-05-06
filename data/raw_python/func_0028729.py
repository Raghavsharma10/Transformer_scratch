def get_results(cmd):
        """
        def get_results(cmd: list) -> str:
            return lines
        Get the ping results using fping.
        :param cmd: List - the fping command and its options
        :return: String - raw string output containing csv fping results
        including the newline characters
        """
        try:
            return subprocess.check_output(cmd)
        except subprocess.CalledProcessError as e:
            return e.output