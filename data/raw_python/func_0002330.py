def _render_uncached_items(self, items, result):
        """
        Render a list of items, that didn't exist in the cache yet.
        """
        for contentitem in items:
            # Render the item.
            # Allow derived classes to skip it.
            try:
                output = self.render_item(contentitem)
            except PluginNotFound as ex:
                result.store_exception(contentitem, ex)
                logger.debug("- item #%s has no matching plugin: %s", contentitem.pk, str(ex))
                continue
            except SkipItem:
                result.set_skipped(contentitem)
                continue

            # Try caching it.
            self._try_cache_output(contentitem, output, result=result)
            if self.edit_mode:
                output.html = markers.wrap_contentitem_output(output.html, contentitem)

            result.store_output(contentitem, output)