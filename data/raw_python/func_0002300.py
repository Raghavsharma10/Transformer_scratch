def can_use_cached_output(self, contentitem):
        """
        Read the cached output - only when search needs it.
        """
        return contentitem.plugin.search_output and not contentitem.plugin.search_fields \
           and super(SearchRenderingPipe, self).can_use_cached_output(contentitem)