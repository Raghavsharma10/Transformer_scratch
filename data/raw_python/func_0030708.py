def stop():
    """
    Stop recording stats.  Call this from a benchmark script when the code you
    want benchmarked has finished.  Call this exactly the same number of times
    you call L{start} and only after calling it.

    @raise RuntimeError: Raised if the parent process responds with anything
    other than an acknowledgement of this message.
    """
    os.write(BenchmarkProcess.BACKCHANNEL_OUT, BenchmarkProcess.STOP)
    response = util.untilConcludes(os.read, BenchmarkProcess.BACKCHANNEL_IN, 1)
    if response != BenchmarkProcess.STOP:
        raise RuntimeError(
            "Parent process responded with %r instead of STOP" % (response,))