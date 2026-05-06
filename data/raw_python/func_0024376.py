def add_backup_operation(self, backup, mode=None):
        """Add a backup operation to the version.

        :param backup: To either add or skip the backup
        :type backup: Boolean
        :param mode: Name of the mode in which the operation is executed
                     For now, backups are mode-independent
        :type mode: String
        """
        try:
            if self.options.backup:
                self.options.backup.ignore_if_operation().execute()
        except OperationError:
            self.backup = backup