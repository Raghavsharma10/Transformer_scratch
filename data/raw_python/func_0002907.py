def with_reactor(*dec_args, **dec_kwargs):
    """
    Decorator for test functions that require a running reactor.
    
    Can be used like this::
    
       @with_reactor
       def test_connect_to_server(self):
          ...
          
    Or like this::
    
       @with_reactor(timeout=10)
       def test_connect_to_server(self):
          ...
          
    If the test function returns a deferred then the test will
    be successful if the deferred resolves to a value or unsuccessful
    if the deferred errbacks.
    
    The test must not leave any connections or a like open. This will
    otherwise result in a reactor-unclean failure of the test.
    
    If there is a function called `twisted_setup()` in the same class
    as the test function is defined, then this function will be invoked
    before the test, but already in the context of the reactor. Note that
    the regular setup function provided by the testing framework will
    be executed too, but not in the reactor context.
    
    Accordingly, if there is a `twisted_teardown()` it executes after the
    test function, even if the test failed. 
    
    If the test, including `twisted_setup` and `twisted_teardown`, has
    not completed within the timout, the test fails. The timeout defaults
    to two minutes. A timeout duration of zero disables the timeout.
    """
    
    # This method takes care of the decorator protocol, it
    # distinguishes between using the decorator with brackets
    # and without brackets. It then calls `_twisted_test_sync()`.

    if len(dec_args) == 1 and callable(dec_args[0]) and not dec_kwargs:
        # decorator used without brackets:
        #   @twisted_test
        #   def test_xxx():
        #     ....
        callee = dec_args[0]
        dec_args = ()
        dec_kwargs = {}
        
        @functools.wraps(callee)
        def wrapper(*call_args, **call_kwargs):
            return _twisted_test_sync(callee, call_args, call_kwargs)
        return wrapper

    else:
        # decorator used with brackets:
        #   @twisted_test(*dec_args, **dec_args)
        #   def test_xxx():
        #     ....
        def decorator(callee):
            @functools.wraps(callee)
            def wrapper(*call_args, **call_kwargs):
                return _twisted_test_sync(callee, call_args, call_kwargs, *dec_args, **dec_kwargs)
            return wrapper
        return decorator