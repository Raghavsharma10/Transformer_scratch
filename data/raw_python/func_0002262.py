def get_output_cache_key(self, placeholder_name, instance):
        """
        .. versionadded:: 0.9
           Return the default cache key which is used to store a rendered item.
           By default, this function generates the cache key using :func:`get_output_cache_base_key`.
        """
        cachekey = self.get_output_cache_base_key(placeholder_name, instance)
        if self.cache_output_per_site:
            cachekey = "{0}-s{1}".format(cachekey, settings.SITE_ID)

        # Append language code
        if self.cache_output_per_language:
            # NOTE: Not using self.language_code, but using the current language instead.
            #       That is what the {% trans %} tags are rendered as after all.
            #       The render_placeholder() code can switch the language if needed.
            user_language = get_language()
            if user_language not in self.cache_supported_language_codes:
                user_language = 'unsupported'
            cachekey = "{0}.{1}".format(cachekey, user_language)

        return cachekey