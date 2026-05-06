def discover_single_case(self, module, case_attributes):
        """Find and load a single TestCase or TestCase method from a module.

        Parameters
        ----------
        module : module
            The imported Python module containing the TestCase to be
            loaded.
        case_attributes : list
            A list (length 1 or 2) of str.  The first component must be
            the name of a TestCase subclass.  The second component must
            be the name of a method in the TestCase.

        """
        # Find single case
        case = module
        loader = self._loader
        for index, component in enumerate(case_attributes):
            case = getattr(case, component, None)
            if case is None:
                return loader.create_suite()
            elif loader.is_test_case(case):
                rest = case_attributes[index + 1:]
                if len(rest) > 1:
                    raise ValueError('Too many components in module path')
                elif len(rest) == 1:
                    return loader.create_suite(
                        [loader.load_test(case, *rest)])
                return loader.load_case(case)

        # No cases matched, return empty suite
        return loader.create_suite()