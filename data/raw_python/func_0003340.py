def Routine(coroutine, scheduler, asyncStart = True, container = None, manualStart = False, daemon = False):
    """
    This wraps a normal coroutine to become a VLCP routine. Usually you do not need to call this yourself;
    `container.start` and `container.subroutine` calls this automatically.
    """
    def run():
        iterator = _await(coroutine)
        iterself = yield
        if manualStart:
            yield
        try:
            if asyncStart:
                scheduler.yield_(iterself)
                yield
            if container is not None:
                container.currentroutine = iterself
            if daemon:
                scheduler.setDaemon(iterself, True)
            try:
                matchers = next(iterator)
            except StopIteration:
                return
            while matchers is None:
                scheduler.yield_(iterself)
                yield
                try:
                    matchers = next(iterator)
                except StopIteration:
                    return
            try:
                scheduler.register(matchers, iterself)
            except Exception:
                try:
                    iterator.throw(IllegalMatchersException(matchers))
                except StopIteration:
                    pass
                raise
            while True:
                try:
                    etup = yield
                except GeneratorExit_:
                    raise
                except:
                    #scheduler.unregister(matchers, iterself)
                    lmatchers = matchers
                    t,v,tr = sys.exc_info()  # @UnusedVariable
                    if container is not None:
                        container.currentroutine = iterself
                    try:
                        matchers = iterator.throw(t,v)
                    except StopIteration:
                        return
                else:
                    #scheduler.unregister(matchers, iterself)
                    lmatchers = matchers
                    if container is not None:
                        container.currentroutine = iterself
                    try:
                        matchers = iterator.send(etup)
                    except StopIteration:
                        return
                while matchers is None:
                    scheduler.yield_(iterself)
                    yield
                    try:
                        matchers = next(iterator)
                    except StopIteration:
                        return
                try:
                    if hasattr(matchers, 'two_way_difference'):
                        reg, unreg = matchers.two_way_difference(lmatchers)
                    else:
                        reg = set(matchers).difference(lmatchers)
                        unreg = set(lmatchers).difference(matchers)
                    scheduler.register(reg, iterself)
                    scheduler.unregister(unreg, iterself)
                except Exception:
                    try:
                        iterator.throw(IllegalMatchersException(matchers))
                    except StopIteration:
                        pass
                    raise
        finally:
            # iterator.close() can be called in other routines, we should restore the currentroutine variable
            if container is not None:
                lastcurrentroutine = getattr(container, 'currentroutine', None)
                container.currentroutine = iterself
            else:
                lastcurrentroutine = None
            _close_generator(coroutine)
            if container is not None:
                container.currentroutine = lastcurrentroutine
            scheduler.unregisterall(iterself)
    r = generatorwrapper(run())
    next(r)
    r.send(r)
    return r