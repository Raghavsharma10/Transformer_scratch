def write_ioc_to_file(self, output_dir=None, force=False):
        """
        Serialize the IOC to a .ioc file.

        :param output_dir: Directory to write the ioc out to.  default is the current working directory.
        :param force: If specified, will not validate the root node of the IOC is 'OpenIOC'.
        :return:
        """
        return write_ioc(self.root, output_dir, force=force)