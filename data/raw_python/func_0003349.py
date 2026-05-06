def with_callback(self, subprocess, callback, *matchers, intercept_callback = None):
        """
        Monitoring event matchers while executing a subprocess. `callback(event, matcher)` is called each time
        an event is matched by any event matchers. If the callback raises an exception, the subprocess is terminated.
        
        :param intercept_callback: a callback called before a event is delegated to the inner subprocess
        """
        it_ = _await(subprocess)
        if not matchers and not intercept_callback:
            return (yield from it_)
        try:
            try:
                m = next(it_)
            except StopIteration as e:
                return e.value
            while True:
                if m is None:
                    try:
                        yield
                    except GeneratorExit_:
                        raise
                    except:
                        t,v,tr = sys.exc_info()  # @UnusedVariable
                        try:
                            m = it_.throw(t,v)
                        except StopIteration as e:
                            return e.value
                    else:                        
                        try:
                            m = next(it_)
                        except StopIteration as e:
                            return e.value
                else:
                    while True:
                        try:
                            ev, matcher = yield m + tuple(matchers)
                        except GeneratorExit_:
                            # subprocess is closed in `finally` clause
                            raise
                        except:
                            # delegate this exception inside
                            t,v,tr = sys.exc_info()  # @UnusedVariable
                            try:
                                m = it_.throw(t,v)
                            except StopIteration as e:
                                return e.value
                        else:
                            if matcher in matchers:
                                callback(ev, matcher)
                            else:
                                if intercept_callback:
                                    intercept_callback(ev, matcher)
                                break
                    try:
                        m = it_.send((ev, matcher))
                    except StopIteration as e:
                        return e.value
        finally:
            _close_generator(subprocess)