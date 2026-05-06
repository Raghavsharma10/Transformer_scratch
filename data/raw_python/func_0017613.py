def group_backups(self, backups):
        """
        Group backups collected by :func:`collect_backups()` by rotation frequencies.

        :param backups: A :class:`set` of :class:`Backup` objects.
        :returns: A :class:`dict` whose keys are the names of rotation
                  frequencies ('hourly', 'daily', etc.) and whose values are
                  dictionaries. Each nested dictionary contains lists of
                  :class:`Backup` objects that are grouped together because
                  they belong into the same time unit for the corresponding
                  rotation frequency.
        """
        backups_by_frequency = dict((frequency, collections.defaultdict(list)) for frequency in SUPPORTED_FREQUENCIES)
        for b in backups:
            backups_by_frequency['minutely'][(b.year, b.month, b.day, b.hour, b.minute)].append(b)
            backups_by_frequency['hourly'][(b.year, b.month, b.day, b.hour)].append(b)
            backups_by_frequency['daily'][(b.year, b.month, b.day)].append(b)
            backups_by_frequency['weekly'][(b.year, b.week)].append(b)
            backups_by_frequency['monthly'][(b.year, b.month)].append(b)
            backups_by_frequency['yearly'][b.year].append(b)
        return backups_by_frequency