def _close_generator(g):
    """
    PyPy 3 generator has a bug that calling `close` caused
    memory leak. Before it is fixed, use `throw` instead
    """
    if isinstance(g, generatorwrapper):
        g.close()
    elif _get_frame(g) is not None:
        try:
            g.throw(GeneratorExit_)
        except (StopIteration, GeneratorExit_):
            return
        else:
            raise RuntimeError("coroutine ignored GeneratorExit")