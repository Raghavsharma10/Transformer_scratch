def prefer_type(self, prefer, over):
        """Prefer one type over another type, all else being equivalent.

        With abstract base classes (Python's abc module) it is possible for
        a type to appear to be a subclass of another type without the supertype
        appearing in the subtype's MRO. As such, the supertype has no order
        with respect to other supertypes, and this may lead to amguity if two
        implementations are provided for unrelated abstract types.

        In such cases, it is possible to disambiguate by explictly telling the
        function to prefer one type over the other.

        Arguments:
            prefer: Preferred type (class).
            over: The type we don't like (class).

        Raises:
            ValueError: In case of logical conflicts.
        """
        self._write_lock.acquire()
        try:
            if self._preferred(preferred=over, over=prefer):
                raise ValueError(
                    "Type %r is already preferred over %r." % (over, prefer))
            prefs = self._prefer_table.setdefault(prefer, set())
            prefs.add(over)
        finally:
            self._write_lock.release()