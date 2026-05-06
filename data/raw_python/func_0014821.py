def __compare_helper(self, other, condition, notimpl_target):
        """Helper for comparison.

        Allows the caller to provide:
        - The condition
        - The return value if the comparison is meaningless (ie versions with
            build metadata).
        """
        if not isinstance(other, self.__class__):
            return NotImplemented

        cmp_res = self.__cmp__(other)
        if cmp_res is NotImplemented:
            return notimpl_target

        return condition(cmp_res)