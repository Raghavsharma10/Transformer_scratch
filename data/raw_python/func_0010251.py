def list_contents(self):
        """List the contents of this directory

        :return: A LsInfo object that contains directories and files
        :rtype: :class:`~.LsInfo` or :class:`~.ErrorInfo`

        Here is an example usage::

            # let dirinfo be a DirectoryInfo object
            ldata = dirinfo.list_contents()
            if isinstance(ldata, ErrorInfo):
                # Do some error handling
                logger.warn("Error listing file info: (%s) %s", ldata.errno, ldata.message)
            # It's of type LsInfo
            else:
                # Look at all the files
                for finfo in ldata.files:
                    logger.info("Found file %s of size %s", finfo.path, finfo.size)
                # Look at all the directories
                for dinfo in ldata.directories:
                    logger.info("Found directory %s of last modified %s", dinfo.path, dinfo.last_modified)
        """
        target = DeviceTarget(self.device_id)
        return self._fssapi.list_files(target, self.path)[self.device_id]