def identify(fn):
        ''' returns a tuple that is used to match
            functions to their neighbors in their
            resident namespaces '''
        return (
            fn.__globals__['__name__'], # module namespace
            getattr(fn, '__qualname__', getattr(fn, '__name__', '')) # class and function namespace
        )
        def __init__(self, fn):
            self.validate_function(fn)
            self.configured = False
            self.has_backup_plan = False
            if self.has_args():
                self.backup_plan = fn
            else:
                self.id = self.identify(fn)
                self.backup_plan = big.overload._cache.get(self.id, None)
                #if self.id in overload._cache:
                #    self.backup_plan =
                self.configure_with(fn)
            #wraps(fn)(self)

        def __call__(self, *args, **kwargs):
            #print(locals())
            try:  # try running like normal
                return self.fn(*args, **kwargs)
            except Exception as ex:
                if self.has_backup_plan:
                    return self.backup_plan(*args, **kwargs) # run backup plan
                elif self.configured:
                    raise ex # no backup plan, abort
                else:
                    # complete unconfigured setup
                    self.configure_with(*args, **kwargs)
                    return self