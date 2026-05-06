def clear_restore_points(self, name=None):
        '''Deleting all/a configuration restore points/point.

        Parameters
        ----------
        name : str
            Name of the restore point to be deleted. If not given, all restore points will be deleted.
        '''
        if name is None:
            self.config_state.clear()
        else:
            del self.config_state[name]