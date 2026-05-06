def object_list(self):
        """
        Return this table's object_list, transformed (sorted, reversed,
        filtered, etc) according to its meta options.
        """

        def _sort(ob, ol):
            reverse = ob.startswith("-")
            ob = ob[1:] if reverse else ob
            for column in self.columns:
                if column.sort_key_fn is not None and column.name == ob:
                    return sorted(ol, key=column.sort_key_fn, reverse=reverse)
            if self._meta.order_by and hasattr(ol, "order_by"):
                return list(ol.order_by(*self._meta.order_by.split("|")))
            return ol

        ol = self._object_list
        ob = self._meta.order_by
        if not ob: return ol
        if isinstance(ob, basestring):
            return _sort(ob, ol)
        elif isinstance(ob, list):
            ob.reverse()
            for fn in ob:
                ol = _sort(fn, ol)
        return ol