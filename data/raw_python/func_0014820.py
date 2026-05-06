def _comparison_functions(cls, partial=False):
        """Retrieve comparison methods to apply on version components.

        This is a private API.

        Args:
            partial (bool): whether to provide 'partial' or 'strict' matching.

        Returns:
            5-tuple of cmp-like functions.
        """

        def prerelease_cmp(a, b):
            """Compare prerelease components.

            Special rule: a version without prerelease component has higher
            precedence than one with a prerelease component.
            """
            if a and b:
                return identifier_list_cmp(a, b)
            elif a:
                # Versions with prerelease field have lower precedence
                return -1
            elif b:
                return 1
            else:
                return 0

        def build_cmp(a, b):
            """Compare build metadata.

            Special rule: there is no ordering on build metadata.
            """
            if a == b:
                return 0
            else:
                return NotImplemented

        def make_optional(orig_cmp_fun):
            """Convert a cmp-like function to consider 'None == *'."""
            @functools.wraps(orig_cmp_fun)
            def alt_cmp_fun(a, b):
                if a is None or b is None:
                    return 0
                return orig_cmp_fun(a, b)

            return alt_cmp_fun

        if partial:
            return [
                base_cmp,  # Major is still mandatory
                make_optional(base_cmp),
                make_optional(base_cmp),
                make_optional(prerelease_cmp),
                make_optional(build_cmp),
            ]
        else:
            return [
                base_cmp,
                base_cmp,
                base_cmp,
                prerelease_cmp,
                build_cmp,
            ]