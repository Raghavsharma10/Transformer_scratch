def render_item(self, contentitem):
        """
        Render the item - but render as search text instead.
        """
        plugin = contentitem.plugin
        if not plugin.search_output and not plugin.search_fields:
            # Only render items when the item was output will be indexed.
            raise SkipItem

        if not plugin.search_output:
            output = ContentItemOutput('', cacheable=False)
        else:
            output = super(SearchRenderingPipe, self).render_item(contentitem)

        if plugin.search_fields:
            # Just add the results into the output, but avoid caching that somewhere.
            output.html += plugin.get_search_text(contentitem)
            output.cacheable = False

        return output