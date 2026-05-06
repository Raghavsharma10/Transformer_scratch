def init_threads(t=None, s=None):
    """Should define dummyThread class and dummySignal class"""
    global THREAD, SIGNAL
    THREAD = t or dummyThread
    SIGNAL = s or dummySignal