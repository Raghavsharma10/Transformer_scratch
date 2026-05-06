def qstat(self, queue_name, return_dict=False):
        """
        Return the status of the queue (currently unimplemented).

        Future support / testing of QSTAT support in Disque

        QSTAT <qname>

        Return produced ... consumed ... idle ... sources [...] ctime ...
        """
        rtn = self.execute_command('QSTAT', queue_name)

        if return_dict:
            grouped = self._grouper(rtn, 2)
            rtn = dict((a, b) for a, b in grouped)

        return rtn