def update_slaves(self):
        """Update all `slave` |Substituter| objects.

        See method |Substituter.update_masters| for further information.
        """
        for slave in self.slaves:
            slave._medium2long.update(self._medium2long)
            slave.update_slaves()