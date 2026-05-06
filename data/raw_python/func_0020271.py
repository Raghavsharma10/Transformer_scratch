def flush(self, exclude=None, include=None, dryrun=False):
        '''Flush :attr:`registered_models`.

        :param exclude: optional list of model names to exclude.
        :param include: optional list of model names to include.
        :param dryrun: Doesn't remove anything, simply collect managers
            to flush.
        :return:
        '''
        exclude = exclude or []
        results = []
        for manager in self._registered_models.values():
            m = manager._meta
            if include is not None and not (m.modelkey in include or
                                            m.app_label in include):
                continue
            if not (m.modelkey in exclude or m.app_label in exclude):
                if dryrun:
                    results.append(manager)
                else:
                    results.append(manager.flush())
        return results