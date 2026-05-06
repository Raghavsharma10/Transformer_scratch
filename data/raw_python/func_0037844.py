def _filter(self, blacklist=None, newest_only=False, **kwargs):
        """
        Args:
            blacklist(tuple): Iterable of of BlacklistEntry objects
            newest_only(bool): Only the newest version of each plugin is returned
            version(str): Specific version to retrieve

        Returns dictionary of plugins

        If a blacklist is supplied, plugins are evaluated against the blacklist entries
        """

        version = kwargs.get('version', None)
        rtn = None

        if self:  # Dict is not empty

            if blacklist:

                blacklist = self._process_blacklist(blacklist)

                if version:
                    if version not in blacklist:
                        rtn = self.get(version, None)

                elif newest_only:
                    for key in reversed(self._sorted_keys()):
                        if key not in blacklist:
                            rtn = self[key]
                            break
                    # If no keys are left, None will be returned
                else:
                    rtn = dict((key, val) for key, val in self.items() if key not in blacklist) \
                          or None

            elif version:
                rtn = self.get(version, None)

            elif newest_only:
                rtn = self[self._sorted_keys()[-1]]

            else:
                rtn = dict(self)

        return rtn