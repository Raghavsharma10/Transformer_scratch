def exec_context(self, **kwargs):
        """Base environment for evals, the stuff that is the same for all evals. Primarily used in the
        Caster pipe"""
        import inspect
        import dateutil.parser
        import datetime
        import random
        from functools import partial
        from ambry.valuetype.types import parse_date, parse_time, parse_datetime
        import ambry.valuetype.types
        import ambry.valuetype.exceptions
        import ambry.valuetype.test
        import ambry.valuetype


        def set_from(f, frm):
            try:
                try:
                    f.ambry_from = frm
                except AttributeError:  # for instance methods
                    f.im_func.ambry_from = frm
            except (TypeError, AttributeError):  # Builtins, non python code
                pass

            return f

        test_env = dict(
            parse_date=parse_date,
            parse_time=parse_time,
            parse_datetime=parse_datetime,
            partial=partial,
            bundle=self
        )

        test_env.update(kwargs)
        test_env.update(dateutil.parser.__dict__)
        test_env.update(datetime.__dict__)
        test_env.update(random.__dict__)
        test_env.update(ambry.valuetype.core.__dict__)
        test_env.update(ambry.valuetype.types.__dict__)
        test_env.update(ambry.valuetype.exceptions.__dict__)
        test_env.update(ambry.valuetype.test.__dict__)
        test_env.update(ambry.valuetype.__dict__)

        localvars = {}

        for f_name, func in test_env.items():
            if not isinstance(func, (str, tuple)):
                localvars[f_name] = set_from(func, 'env')

        # The 'b' parameter of randint is assumed to be a bundle, but
        # replacing it with a lambda prevents the param assignment
        localvars['randint'] = lambda a, b: random.randint(a, b)

        if self != Bundle:
            # Functions from the bundle
            base = set(inspect.getmembers(Bundle, predicate=inspect.isfunction))
            mine = set(inspect.getmembers(self.__class__, predicate=inspect.isfunction))

            localvars.update({f_name: set_from(func, 'bundle') for f_name, func in mine - base})

            # Bound methods. In python 2, these must be called referenced from the bundle, since
            # there is a difference between bound and unbound methods. In Python 3, there is no differnce,
            # so the lambda functions may not be necessary.
            base = set(inspect.getmembers(Bundle, predicate=inspect.ismethod))
            mine = set(inspect.getmembers(self.__class__, predicate=inspect.ismethod))

            # Functions are descriptors, and the __get__ call binds the function to its object to make a bound method
            localvars.update({f_name: set_from(func.__get__(self), 'bundle') for f_name, func in (mine - base)})

        # Bundle module functions

        module_entries = inspect.getmembers(sys.modules['ambry.build'], predicate=inspect.isfunction)


        localvars.update({f_name: set_from(func, 'module') for f_name, func in module_entries})

        return localvars