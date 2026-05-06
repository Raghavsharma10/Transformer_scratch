def pre_freeze_hook(self):
        """Pre :meth:`dtoolcore.ProtoDataSet.freeze` actions.

        This method is called at the beginning of the
        :meth:`dtoolcore.ProtoDataSet.freeze` method.

        It may be useful for remote storage backends to generate
        caches to remove repetitive time consuming calls
        """
        allowed = set([v[0] for v in _STRUCTURE_PARAMETERS.values()])
        for d in os.listdir(self._abspath):
            if d not in allowed:
                msg = "Rogue content in base of dataset: {}".format(d)
                raise(DiskStorageBrokerValidationWarning(msg))