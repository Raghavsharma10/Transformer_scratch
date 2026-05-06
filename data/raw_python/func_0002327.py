def render_items(self, placeholder, items, parent_object=None, template_name=None, cachable=None):
        """
        The main rendering sequence.
        """
        # Unless it was done before, disable polymorphic effects.
        is_queryset = False
        if hasattr(items, "non_polymorphic"):
            is_queryset = True
            if not items.polymorphic_disabled and items._result_cache is None:
                items = items.non_polymorphic()

        # See if the queryset contained anything.
        # This test is moved here, to prevent earlier query execution.
        if not items:
            logger.debug("- no items in placeholder '%s'", get_placeholder_debug_name(placeholder))
            return ContentItemOutput(mark_safe(u"<!-- no items in placeholder '{0}' -->".format(escape(get_placeholder_name(placeholder)))), cacheable=True)

        # Tracked data during rendering:
        result = self.result_class(
            request=self.request,
            parent_object=parent_object,
            placeholder=placeholder,
            items=items,
            all_cacheable=self._can_cache_merged_output(template_name, cachable),
        )
        if self.edit_mode:
            result.set_uncachable()

        if is_queryset:
            # Phase 1: get cached output
            self._fetch_cached_output(items, result=result)
            result.fetch_remaining_instances()
        else:
            # The items is either a list of manually created items, or it's a QuerySet.
            # Can't prevent reading the subclasses only, so don't bother with caching here.
            result.add_remaining_list(items)

        # Start the actual rendering of remaining items.
        if result.remaining_items:
            # Phase 2: render remaining items
            self._render_uncached_items(result.remaining_items, result=result)

        # And merge all items together.
        return self.merge_output(result, items, template_name)