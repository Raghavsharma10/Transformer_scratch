def _refresh_hierarchy_recursive(self, cached_hierarchy, file_hierarchy):
        """Recursively goes through given corresponding hierarchies from cache and filesystem
        and adds/refreshes/removes added/changed/removed assistants.

        Args:
            cached_hierarchy: the respective hierarchy part from current cache
                              (for format see Cache class docstring)
            file_hierarchy: the respective hierarchy part from filesystem
                            (for format see what refresh_role accepts)

        Returns:
            True if self.cache has been changed, False otherwise (doesn't write anything
            to cache file)
        """
        was_change = False
        cached_ass = set(cached_hierarchy.keys())
        new_ass = set(file_hierarchy.keys())

        to_add = new_ass - cached_ass
        to_remove = cached_ass - new_ass
        to_check = cached_ass - to_remove

        if to_add or to_remove:
            was_change = True

        for ass in to_add:
            cached_hierarchy[ass] = self._new_ass_hierarchy(file_hierarchy[ass])

        for ass in to_remove:
            del cached_hierarchy[ass]

        for ass in to_check:
            needs_refresh = False
            try:
                needs_refresh = self._ass_needs_refresh(cached_hierarchy[ass], file_hierarchy[ass])
            except:
                needs_refresh = True

            if needs_refresh:
                self._ass_refresh_attrs(cached_hierarchy[ass], file_hierarchy[ass])
                was_change = True
            was_change |= self._refresh_hierarchy_recursive(
                cached_hierarchy[ass]['subhierarchy'],
                file_hierarchy[ass]['subhierarchy'])

        return was_change