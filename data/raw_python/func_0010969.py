def set_debug_listener(stream):
    """Break into a debugger if receives the SIGUSR1 signal"""

    def debugger(sig, frame):
        launch_debugger(frame, stream)

    if hasattr(signal, 'SIGUSR1'):
        signal.signal(signal.SIGUSR1, debugger)
    else:
        logger.warn("Cannot set SIGUSR1 signal for debug mode.")