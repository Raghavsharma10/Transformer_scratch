def write_pruned_iocs(self, directory=None, pruned_source=None):
        """
        Writes IOCs to a directory that have been pruned of some or all IOCs.

        :param directory: Directory to write IOCs to.  If not provided, the current working directory is used.
        :param pruned_source: Iterable containing a set of iocids.  Defaults to self.iocs_10.
        :return:
        """
        """
        write_pruned_iocs to a directory

        if directory is None, write the iocs to the current working directory
        """
        if pruned_source is None:
            pruned_source = self.pruned_11_iocs
        if len(pruned_source) < 1:
            log.error('no iocs available to write out')
            return False
        if not directory:
            directory = os.getcwd()
        if os.path.isfile(directory):
            log.error('cannot writes iocs to a directory')
            return False
        utils.safe_makedirs(directory)
        output_dir = os.path.abspath(directory)
        # serialize the iocs
        for iocid in pruned_source:
            ioc_obj = self.iocs_10[iocid]
            ioc_obj.write_ioc_to_file(output_dir=output_dir, force=True)
        return True