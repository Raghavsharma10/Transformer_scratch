def rotate_backups(self, location, load_config=True, prepare=False):
        """
        Rotate the backups in a directory according to a flexible rotation scheme.

        :param location: Any value accepted by :func:`coerce_location()`.
        :param load_config: If :data:`True` (so by default) the rotation scheme
                            and other options can be customized by the user in
                            a configuration file. In this case the caller's
                            arguments are only used when the configuration file
                            doesn't define a configuration for the location.
        :param prepare: If this is :data:`True` (not the default) then
                        :func:`rotate_backups()` will prepare the required
                        rotation commands without running them.
        :returns: A list with the rotation commands
                  (:class:`~executor.ExternalCommand` objects).
        :raises: :exc:`~exceptions.ValueError` when the given location doesn't
                 exist, isn't readable or isn't writable. The third check is
                 only performed when dry run isn't enabled.

        This function binds the main methods of the :class:`RotateBackups`
        class together to implement backup rotation with an easy to use Python
        API. If you're using `rotate-backups` as a Python API and the default
        behavior is not satisfactory, consider writing your own
        :func:`rotate_backups()` function based on the underlying
        :func:`collect_backups()`, :func:`group_backups()`,
        :func:`apply_rotation_scheme()` and
        :func:`find_preservation_criteria()` methods.
        """
        rotation_commands = []
        location = coerce_location(location)
        # Load configuration overrides by user?
        if load_config:
            location = self.load_config_file(location)
        # Collect the backups in the given directory.
        sorted_backups = self.collect_backups(location)
        if not sorted_backups:
            logger.info("No backups found in %s.", location)
            return
        # Make sure the directory is writable.
        if not self.dry_run:
            location.ensure_writable()
        most_recent_backup = sorted_backups[-1]
        # Group the backups by the rotation frequencies.
        backups_by_frequency = self.group_backups(sorted_backups)
        # Apply the user defined rotation scheme.
        self.apply_rotation_scheme(backups_by_frequency, most_recent_backup.timestamp)
        # Find which backups to preserve and why.
        backups_to_preserve = self.find_preservation_criteria(backups_by_frequency)
        # Apply the calculated rotation scheme.
        for backup in sorted_backups:
            friendly_name = backup.pathname
            if not location.is_remote:
                # Use human friendly pathname formatting for local backups.
                friendly_name = format_path(backup.pathname)
            if backup in backups_to_preserve:
                matching_periods = backups_to_preserve[backup]
                logger.info("Preserving %s (matches %s retention %s) ..",
                            friendly_name, concatenate(map(repr, matching_periods)),
                            "period" if len(matching_periods) == 1 else "periods")
            else:
                logger.info("Deleting %s ..", friendly_name)
                if not self.dry_run:
                    # Copy the list with the (possibly user defined) removal command.
                    removal_command = list(self.removal_command)
                    # Add the pathname of the backup as the final argument.
                    removal_command.append(backup.pathname)
                    # Construct the command object.
                    command = location.context.prepare(
                        command=removal_command,
                        group_by=(location.ssh_alias, location.mount_point),
                        ionice=self.io_scheduling_class,
                    )
                    rotation_commands.append(command)
                    if not prepare:
                        timer = Timer()
                        command.wait()
                        logger.verbose("Deleted %s in %s.", friendly_name, timer)
        if len(backups_to_preserve) == len(sorted_backups):
            logger.info("Nothing to do! (all backups preserved)")
        return rotation_commands