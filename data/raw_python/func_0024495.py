def _dump_cache(self, _shifts):
        """
        _shifts dumps in /tmp directory after reboot it will drop
        """
        with open(self.__cache_path, 'wb') as f:
            dump(_shifts, f)