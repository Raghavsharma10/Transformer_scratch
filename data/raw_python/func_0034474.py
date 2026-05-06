def enable_thread_profiling(profile_dir, exception_callback=None):
    """
    Monkey-patch the threading.Thread class with our own ProfiledThread. Any subsequent imports of threading.Thread
    will reference ProfiledThread instead.
    """
    global profiled_thread_enabled, Thread, Process
    if os.path.isdir(profile_dir):
        _Profiler.profile_dir = profile_dir
    else:
        raise OSError('%s does not exist' % profile_dir)
    _Profiler.exception_callback = exception_callback
    Thread = threading.Thread = ProfiledThread
    Process = multiprocessing.Process = ProfiledProcess
    profiled_thread_enabled = True