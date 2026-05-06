def get_output_cache_keys(self, placeholder_name, instance):
        """
        .. versionadded:: 0.9
           Return the possible cache keys for a rendered item.

           This method should be overwritten when implementing a function :func:`set_cached_output` method
           or when implementing a :func:`get_output_cache_key` function.
           By default, this function generates the cache key using :func:`get_output_cache_base_key`.
        """
        base_key = self.get_output_cache_base_key(placeholder_name, instance)
        cachekeys = [
            base_key,
        ]

        if self.cache_output_per_site:
            site_ids = list(Site.objects.values_list('pk', flat=True))
            if settings.SITE_ID not in site_ids:
                site_ids.append(settings.SITE_ID)

            base_key = get_rendering_cache_key(placeholder_name, instance)
            cachekeys = ["{0}-s{1}".format(base_key, site_id) for site_id in site_ids]

        if self.cache_output_per_language or self.render_ignore_item_language:
            # Append language code to all keys,
            # have to invalidate a lot more items in memcache.
            # Also added "None" suffix, since get_parent_language_code() may return that.
            # TODO: ideally for render_ignore_item_language, only invalidate all when the fallback language changed.
            total_list = []
            cache_languages = list(self.cache_supported_language_codes) + ['unsupported', 'None']

            # All variants of the Placeholder (for full page caching)
            placeholder = instance.placeholder
            total_list.extend(get_placeholder_cache_key(placeholder, lc) for lc in cache_languages)

            # All variants of the ContentItem in different languages
            for user_language in cache_languages:
                total_list.extend("{0}.{1}".format(base, user_language) for base in cachekeys)
            cachekeys = total_list

        return cachekeys