def _search(self, category, name, recurse=True):
        """Search the scope stack for the name in the specified
        category (types/locals/vars).

        :category: the category to search in (locals/types/vars)
        :name: name to search for
        :returns: None if not found, the result of the found local/type/id
        """
        idx = len(self._scope_stack) - 1
        curr = self._curr_scope
        for scope in reversed(self._scope_stack):
            res = scope[category].get(name, None)
            if res is not None:
                return res

        if recurse and self._parent is not None:
            return self._parent._search(category, name, recurse)

        return None