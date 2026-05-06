def init_log_rate(output_f, N=None, message='', print_rate=None):
    """Initialze the log_rate function. Returnas a partial function to call for
    each event.

    If N is not specified but print_rate is specified, the initial N is
    set to 100, and after the first message, the N value is adjusted to
    emit print_rate messages per second

    """

    if print_rate and not N:
        N = 100

    if not N:
        N = 5000

    d = [0,  # number of items processed
         time(),  # start time. This one gets replaced after first message
         N,  # ticker to next message
         N,  # frequency to log a message
         message,
         print_rate,
         deque([], maxlen=4)  # Deque for averaging last N rates
         ]

    assert isinstance(output_f, Callable)

    f = partial(_log_rate, output_f, d)
    f.always = output_f
    f.count = lambda: d[0]

    return f