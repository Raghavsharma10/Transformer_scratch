def restore(self, backup=None, delete_backup=False):
        """Restore the snapshot to the associated storage resource.

        :param backup: name of the backup snapshot
        :param delete_backup: Whether to delete the backup snap after a
                              successful restore.
        """
        resp = self._cli.action(self.resource_class, self.get_id(),
                                'restore', copyName=backup)
        resp.raise_if_err()
        backup = resp.first_content['backup']
        backup_snap = UnitySnap(_id=backup['id'], cli=self._cli)

        if delete_backup:
            log.info("Deleting the backup snap {} as the restoration "
                     "succeeded.".format(backup['id']))
            backup_snap.delete()

        return backup_snap