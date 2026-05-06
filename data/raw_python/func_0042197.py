def invalidate_cache(self, obj=None, queryset=None,
                         extra=None, force_all=False):
        """
        Method that should be called by all tiggers to invalidate the
        cache for an item(s).

        Should be overriden by inheriting classes to customize behavior.
        """

        if self.cache_manager:
            if queryset != None:
                force_all = True

            self.cache_manager.invalidate_cache(self.model, instance=obj,
                                                   extra=extra,
                                                   force_all=force_all)