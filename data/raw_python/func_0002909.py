def _timeoutDeferred(deferred, timeout):
    """
    Cancels the given deferred after the given time, if it has not yet callbacked/errbacked it.
    """
    delayedCall = reactor.callLater(timeout, deferred.cancel)
    def gotResult(result):
        if delayedCall.active():
            delayedCall.cancel()
        return result
    deferred.addBoth(gotResult)