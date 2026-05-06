def allocator_disabled(self):
        """Return a simplified one-word answer to the question, 'Has the 
        automatic shard allocator been disabled for this cluster?'

        The answer will be one of "disabled" (yes), "enabled" (no), or 
        "unknown".

        """
        state = "unknown"
        setting_getters = [
            lambda s: s['cluster.routing.allocation.disable_allocation'],
            lambda s: s['cluster']['routing']['allocation']['disable_allocation']]
        settings = self.get('/_cluster/settings')
        for i in ['persistent', 'transient']:
            for getter in setting_getters:
                try:
                    v = booleanise(getter(settings[i]))
                    if v == True:
                        state = "disabled"
                    elif v == False:
                        state = "enabled"
                except KeyError:
                    pass
        return state