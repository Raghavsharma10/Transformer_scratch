def configurations(self):
        """Configurations from uwsgiconf module."""

        if self._confs is not None:
            return self._confs

        with output_capturing():
            module = self.load(self.fpath)
            confs = getattr(module, CONFIGS_MODULE_ATTR)
            confs = listify(confs)

        self._confs = confs

        return confs