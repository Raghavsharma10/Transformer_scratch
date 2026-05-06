def has_any():
    "Returns the best available proactor implementation for the current platform."
    return get_first(has_ctypes_iocp, has_iocp, has_stdlib_kqueue, has_kqueue, 
                        has_stdlib_epoll, has_epoll, has_poll, has_select)