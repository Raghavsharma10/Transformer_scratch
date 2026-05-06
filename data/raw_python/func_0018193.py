def atfork(prepare=None, parent=None, child=None):
    """A Python work-a-like of pthread_atfork.
    
    Any time a fork() is called from Python, all 'prepare' callables will
    be called in the order they were registered using this function.

    After the fork (successful or not), all 'parent' callables will be called in
    the parent process.  If the fork succeeded, all 'child' callables will be
    called in the child process.

    No exceptions may be raised from any of the registered callables.  If so
    they will be printed to sys.stderr after the fork call once it is safe
    to do so.
    """
    assert not prepare or callable(prepare)
    assert not parent or callable(parent)
    assert not child or callable(child)
    _fork_lock.acquire()
    try:
        if prepare:
            _prepare_call_list.append(prepare)
        if parent:
            _parent_call_list.append(parent)
        if child:
            _child_call_list.append(child)
    finally:
        _fork_lock.release()