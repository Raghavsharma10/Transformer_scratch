def collect_backups(self, location):
        """
        Collect the backups at the given location.

        :param location: Any value accepted by :func:`coerce_location()`.
        :returns: A sorted :class:`list` of :class:`Backup` objects (the
                  backups are sorted by their date).
        :raises: :exc:`~exceptions.ValueError` when the given directory doesn't
                 exist or isn't readable.
        """
        backups = []
        location = coerce_location(location)
        logger.info("Scanning %s for backups ..", location)
        location.ensure_readable()
        for entry in natsort(location.context.list_entries(location.directory)):
            match = TIMESTAMP_PATTERN.search(entry)
            if match:
                if self.exclude_list and any(fnmatch.fnmatch(entry, p) for p in self.exclude_list):
                    logger.verbose("Excluded %s (it matched the exclude list).", entry)
                elif self.include_list and not any(fnmatch.fnmatch(entry, p) for p in self.include_list):
                    logger.verbose("Excluded %s (it didn't match the include list).", entry)
                else:
                    try:
                        backups.append(Backup(
                            pathname=os.path.join(location.directory, entry),
                            timestamp=datetime.datetime(*(int(group, 10) for group in match.groups('0'))),
                        ))
                    except ValueError as e:
                        logger.notice("Ignoring %s due to invalid date (%s).", entry, e)
            else:
                logger.debug("Failed to match time stamp in filename: %s", entry)
        if backups:
            logger.info("Found %i timestamped backups in %s.", len(backups), location)
        return sorted(backups)