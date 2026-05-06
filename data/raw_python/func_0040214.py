def process_kill(pid, sig=None):
    """Send signal to process.
    """
    sig = sig or signal.SIGTERM
    os.kill(pid, sig)