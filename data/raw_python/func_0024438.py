def _parse_options(self, migration):
        """Build :class:`MigrationOption` and
        :class:`MigrationBackupOption` instances."""
        options = migration.get('options', {})
        install_command = options.get('install_command')
        backup = options.get('backup')
        if backup:
            self.check_dict_expected_keys(
                {'command', 'ignore_if', 'stop_on_failure'},
                options['backup'], 'backup',
            )
            backup = MigrationBackupOption(
                command=backup.get('command'),
                ignore_if=backup.get('ignore_if'),
                stop_on_failure=backup.get('stop_on_failure', True),
            )
        return MigrationOption(
            install_command=install_command,
            backup=backup,
        )