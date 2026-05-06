def _ensure_reactor_running():
    """
    Starts the twisted reactor if it is not running already.
    
    The reactor is started in a new daemon-thread.
    
    Has to perform dirty hacks so that twisted can register
    signals even if it is not running in the main-thread.
    """
    if not reactor.running:
        
        # Some of the `signal` API can only be called
        # from the main-thread. So we do a dirty workaround.
        #
        # `signal.signal()` and `signal.wakeup_fd_capture()`
        # are temporarily monkey-patched while the reactor is
        # starting.
        #
        # The patched functions record the invocations in
        # `signal_registrations`. 
        #
        # Once the reactor is started, the main-thread
        # is used to playback the recorded invocations.
        
        signal_registrations = []

        # do the monkey patching
        def signal_capture(*args, **kwargs):
            signal_registrations.append((orig_signal, args, kwargs))
        def set_wakeup_fd_capture(*args, **kwargs):
            signal_registrations.append((orig_set_wakeup_fd, args, kwargs))
        orig_signal = signal.signal
        signal.signal = signal_capture
        orig_set_wakeup_fd = signal.set_wakeup_fd
        signal.set_wakeup_fd = set_wakeup_fd_capture
        
        
        # start the reactor in a daemon-thread
        reactor_thread = threading.Thread(target=reactor.run, name="reactor")
        reactor_thread.daemon = True
        reactor_thread.start()
        while not reactor.running:
            time.sleep(0.01)
            
        # Give the reactor a moment to register the signals. 
        # Apparently the 'running' flag is set before that.
        time.sleep(0.01)
        
        # Undo the monkey-paching
        signal.signal = orig_signal
        signal.set_wakeup_fd = orig_set_wakeup_fd
        
        # Playback the recorded calls
        for func, args, kwargs in signal_registrations:
            func(*args, **kwargs)