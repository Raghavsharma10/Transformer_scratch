def parent_after_fork_release():
    """
    Call all parent after fork callables, release the lock and print
    all prepare and parent callback exceptions.
    """
    prepare_exceptions = list(_prepare_call_exceptions)
    del _prepare_call_exceptions[:]
    exceptions = _call_atfork_list(_parent_call_list)
    _fork_lock.release()
    _print_exception_list(prepare_exceptions, 'before fork')
    _print_exception_list(exceptions, 'after fork from parent')