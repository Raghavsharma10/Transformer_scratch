def _support_op(*args):
        """Internal decorator to define an criteria compare operations."""
        def inner(func):
            for one_arg in args:
                _op_mapping_[one_arg] = func
            return func

        return inner