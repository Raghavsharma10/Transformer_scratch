def _fetch_cached_output(self, items, result):
        """
        First try to fetch all items from the cache.
        The items are 'non-polymorphic', so only point to their base class.
        If these are found, there is no need to query the derived data from the database.
        """
        if not appsettings.FLUENT_CONTENTS_CACHE_OUTPUT or not self.use_cached_output:
            result.add_remaining_list(items)
            return

        for contentitem in items:
            result.add_ordering(contentitem)
            output = None

            try:
                plugin = contentitem.plugin
            except PluginNotFound as ex:
                result.store_exception(contentitem, ex)  # Will deal with that later.
                logger.debug("- item #%s has no matching plugin: %s", contentitem.pk, str(ex))
                continue

            # Respect the cache output setting of the plugin
            if self.can_use_cached_output(contentitem):
                result.add_plugin_timeout(plugin)
                output = plugin.get_cached_output(result.placeholder_name, contentitem)

                # Support transition to new output format.
                if output is not None and not isinstance(output, ContentItemOutput):
                    output = None
                    logger.debug("Flushed cached output of {0}#{1} to store new ContentItemOutput format (key: {2})".format(
                        plugin.type_name,
                        contentitem.pk,
                        get_placeholder_name(contentitem.placeholder)
                    ))

            # For debugging, ignore cached values when the template is updated.
            if output and settings.DEBUG:
                cachekey = get_rendering_cache_key(result.placeholder_name, contentitem)
                if is_template_updated(self.request, contentitem, cachekey):
                    output = None

            if output:
                result.store_output(contentitem, output)
            else:
                result.add_remaining(contentitem)