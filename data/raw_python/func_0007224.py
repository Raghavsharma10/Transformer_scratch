def clear(self):
        """Clear the data, and thus, force it to be created on the next fetch.  This is
        done by removing the attribute from ``owner``, deleting it from globals
        and removing the file from the disk.

        """
        vname = self.varname
        if self.path.exists():
            logger.debug('deleting cached work: {}'.format(self.path))
            self.path.unlink()
        if self.owner is not None and hasattr(self.owner, vname):
            logger.debug('removing instance var: {}'.format(vname))
            delattr(self.owner, vname)
        self.clear_global()