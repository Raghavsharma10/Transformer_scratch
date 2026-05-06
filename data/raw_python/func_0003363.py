def depend(*args):
    """
    Decorator to declare dependencies to other modules. Recommended usage is::
    
        import other_module
        
        @depend(other_module.ModuleClass)
        class MyModule(Module):
            ...
    
    :param \*args: depended module classes.
    """
    def decfunc(cls):
        if not 'depends' in cls.__dict__:
            cls.depends = []
        cls.depends.extend(list(args))
        for a in args:
            if not hasattr(a, 'referencedBy'):
                a.referencedBy = []
            a.referencedBy.append(cls)
        return cls
    return decfunc