def get_index_translog_disable_flush(self):
        """Return a dictionary showing the position of the 
        'translog.disable_flush' knob for each index in the cluster.

        The dictionary will look like this:

            {
                "index1": True,      # Autoflushing DISABLED
                "index2": False,     # Autoflushing ENABLED
                "index3": "unknown", # Using default setting (probably enabled)
                ...
            }

        """
        disabled = {}
        settings = self.get('/_settings')
        setting_getters = [
            lambda s: s['index.translog.disable_flush'],
            lambda s: s['index']['translog']['disable_flush']]
        for idx in settings:
            idx_settings = settings[idx]['settings']
            for getter in setting_getters:
                try:
                    disabled[idx] = booleanise(getter(idx_settings))
                except KeyError as e:
                    pass
            if not idx in disabled:
                disabled[idx] = 'unknown'
        return disabled