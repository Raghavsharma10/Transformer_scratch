def get_wf_from_path(self, path):
        """
        load xml from given path
        Args:
            path: diagram path

        Returns:

        """
        with open(path) as fp:
            content = fp.read()
        return [(os.path.basename(os.path.splitext(path)[0]), content), ]