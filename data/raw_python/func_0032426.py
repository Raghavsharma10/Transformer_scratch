def pipeline(pipe=None, name=None, autoexec=False, exit_handler=None):
    """
    This is the foundational function for all of redpipe.
    Everything goes through here.
    create pipelines, nest pipelines, get pipelines for a specific name.
    It all happens here.

    Here's a simple example:

    .. code:: python

        with pipeline() as pipe:
            pipe.set('foo', 'bar')
            foo = pipe.get('foo')
            pipe.execute()
        print(foo)
        > bar

    Now let's look at how we can nest a pipeline.

    .. code:: python

        def process(key, pipe=None):
            with pipeline(pipe, autoexec=True) as pipe:
                return pipe.incr(key)

        with pipeline() as pipe:
            key1 = process('key1', pipe)
            key2 = process('key2', pipe)
            pipe.execute()

        print([key1, key2])

        > [1, 1]


    :param pipe: a Pipeline() or NestedPipeline() object, or None
    :param name: str, optional. the name of the connection to use.
    :param autoexec: bool, if true, implicitly execute the pipe
    :return: Pipeline or NestedPipeline
    """
    if pipe is None:
        return Pipeline(name=name, autoexec=autoexec,
                        exit_handler=exit_handler)

    try:
        if pipe.supports_redpipe_pipeline():
            return NestedPipeline(
                parent=pipe,
                name=name,
                autoexec=autoexec,
                exit_handler=exit_handler
            )
    except AttributeError:
        pass

    raise InvalidPipeline('check your configuration')