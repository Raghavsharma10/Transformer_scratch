def can_use_cached_output(self, contentitem):
        """
        Tell whether the code should try reading cached output
        """
        plugin = contentitem.plugin
        return appsettings.FLUENT_CONTENTS_CACHE_OUTPUT and plugin.cache_output and contentitem.pk