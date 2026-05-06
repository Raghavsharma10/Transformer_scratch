def start():
    """
    Start recording stats.  Call this from a benchmark script when your setup
    is done.  Call this at most once.

    @raise RuntimeError: Raised if the parent process responds with anything
    other than an acknowledgement of this message.
    """
    os.write(BenchmarkProcess.BACKCHANNEL_OUT, BenchmarkProcess.START)
    response = util.untilConcludes(os.read, BenchmarkProcess.BACKCHANNEL_IN, 1)
    if response != BenchmarkProcess.START:
        raise RuntimeError(
            "Parent process responded with %r instead of START " % (response,))