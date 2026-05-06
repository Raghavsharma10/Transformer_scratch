def render_placeholder(self, placeholder, parent_object=None, template_name=None, cachable=None, limit_parent_language=True, fallback_language=None):
        """
        The main rendering sequence for placeholders.
        This will do all the magic for caching, and call :func:`render_items` in the end.
        """
        placeholder_name = get_placeholder_debug_name(placeholder)
        logger.debug("Rendering placeholder '%s'", placeholder_name)

        # Determine whether the placeholder can be cached.
        cachable = self._can_cache_merged_output(template_name, cachable)
        try_cache = cachable and self.may_cache_placeholders()
        logger.debug("- try_cache=%s cachable=%s template_name=%s", try_cache, cachable, template_name)

        if parent_object is None:
            # To support filtering the placeholders by parent language, the parent object needs to be known.
            # Fortunately, the PlaceholderFieldDescriptor makes sure this doesn't require an additional query.
            parent_object = placeholder.parent

        # Fetch the placeholder output from cache.
        language_code = get_parent_language_code(parent_object)
        cache_key = None
        output = None
        if try_cache:
            cache_key = get_placeholder_cache_key_for_parent(parent_object, placeholder.slot, language_code)
            output = cache.get(cache_key)
            if output:
                logger.debug("- fetched cached output")

        if output is None:
            # Get the items, and render them
            items, is_fallback = self._get_placeholder_items(placeholder, parent_object, limit_parent_language, fallback_language, try_cache)
            output = self.render_items(placeholder, items, parent_object, template_name, cachable)

            if is_fallback:
                # Caching fallbacks is not supported yet,
                # content could be rendered in a different gettext language domain.
                output.cacheable = False

            # Store the full-placeholder contents in the cache.
            if try_cache and output.cacheable:
                if output.cache_timeout is not DEFAULT_TIMEOUT:
                    # The timeout is based on the minimal timeout used in plugins.
                    cache.set(cache_key, output, output.cache_timeout)
                else:
                    # Don't want to mix into the default 0/None issue.
                    cache.set(cache_key, output)

        return output