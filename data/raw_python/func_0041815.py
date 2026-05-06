def self_aware(fn):
    ''' decorating a function with this allows it to 
        refer to itself as 'self' inside the function
        body.
    '''
    if isgeneratorfunction(fn):
        @wraps(fn)
        def wrapper(*a,**k):
            generator = fn(*a,**k)
            if hasattr(
                generator, 
                'gi_frame'
            ) and hasattr(
                generator.gi_frame, 
                'f_builtins'
            ) and hasattr(
                generator.gi_frame.f_builtins, 
                '__setitem__'
            ):
                generator.gi_frame.f_builtins[
                    'self'
                ] = generator
        return wrapper
    else:
        fn=strict_globals(**fn.__globals__)(fn)
        fn.__globals__['self']=fn
        return fn