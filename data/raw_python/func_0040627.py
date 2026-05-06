def respond_to_SIGTERM(signal_number, frame, target=None):
    """ these classes are instrumented to respond to a KeyboardInterrupt by
    cleanly shutting down.  This function, when given as a handler to for
    a SIGTERM event, will make the program respond to a SIGTERM as neatly
    as it responds to ^C.

    This function is used in registering a signal handler from the signal
    module.  It should be registered for any signal for which the desired
    behavior is to kill the application:
        signal.signal(signal.SIGTERM, respondToSIGTERM)
        signal.signal(signal.SIGHUP, respondToSIGTERM)

    parameters:
        signal_number - unused in this function but required by the api.
        frame - unused in this function but required by the api.
        target - an instance of a class that has a member called 'task_manager'
                 that is a derivative of the TaskManager class below.
    """
    if target:
        target.config.logger.info('detected SIGTERM')
        # by setting the quit flag to true, any calls to the 'quit_check'
        # method that is so liberally passed around in this framework will
        # result in raising the quit exception.  The current quit exception
        # is KeyboardInterrupt
        target.task_manager.quit = True
    else:
        raise KeyboardInterrupt