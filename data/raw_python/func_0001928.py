def write_iocs(self, directory=None, source=None):
        """
        Serializes IOCs to a directory.

        :param directory: Directory to write IOCs to.  If not provided, the current working directory is used.
        :param source: Dictionary contianing iocid -> IOC mapping.  Defaults to self.iocs_10. This is not normally modifed by a user for this class.
        :return:
        """
        """


        if directory is None, write the iocs to the current working directory
        source: allows specifying a different dictionry of elmentTree ioc objects
        """
        if not source:
            source = self.iocs_10
        if len(source) < 1:
            log.error('no iocs available to write out')
            return False
        if not directory:
            directory = os.getcwd()
        if os.path.isfile(directory):
            log.error('cannot writes iocs to a directory')
            return False
        source_iocs = set(source.keys())
        source_iocs = source_iocs.difference(self.pruned_11_iocs)
        source_iocs = source_iocs.difference(self.null_pruned_iocs)
        if not source_iocs:
            log.error('no iocs available to write out after removing pruned/null iocs')
            return False
        utils.safe_makedirs(directory)
        output_dir = os.path.abspath(directory)
        log.info('Writing IOCs to %s' % (str(output_dir)))
        # serialize the iocs
        for iocid in source_iocs:
            ioc_obj = source[iocid]
            ioc_obj.write_ioc_to_file(output_dir=output_dir, force=True)
        return True