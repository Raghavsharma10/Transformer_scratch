def with_none(self):
        """Convert the NameQuery.NONE to None. This is needed because on the
        kwargs list, a None value means the field is not specified, which
        equates to ANY. The _find_orm() routine, however, is easier to write if
        the NONE value is actually None.

        Returns a clone of the origin, with NONE converted to None

        """

        n = self.clone()

        for k, _, _ in n.name_parts:

            if getattr(n, k) == n.NONE:
                delattr(n, k)

        n.use_clear_dict = False

        return n