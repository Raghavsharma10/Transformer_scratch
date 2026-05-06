def restore(self, res_id, backup_snap=None):
        """
        Restores a snapshot.
        :param res_id: the LUN number of primary LUN or snapshot mount point to
            be restored.
        :param backup_snap: the name of a backup snapshot to be created before
            restoring.
        """
        name = self._get_name()
        out = self._cli.restore_snap(name, res_id, backup_snap)
        ex.raise_if_err(out, 'failed to restore snap {}.'.format(name),
                        default=ex.VNXSnapError)