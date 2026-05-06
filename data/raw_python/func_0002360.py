def _unwindGenerator(self, generator, _prev=None):
        """Unwind (resume) generator."""
        while True:
            if _prev:
                ret, _prev = _prev, None
            else:
                try:
                    ret = next(generator)
                except StopIteration:
                    break

            if isinstance(ret, Request):
                if ret.callback:
                    warnings.warn("Got a request with callback set, bypassing "
                                  "the generator wrapper. Generator may not "
                                  "be able to resume. %s" % ret)
                elif ret.errback:
                    # By Scrapy defaults, a request without callback defaults to
                    # self.parse spider method.
                    warnings.warn("Got a request with errback set, bypassing "
                                  "the generator wrapper. Generator may not "
                                  "be able to resume. %s" % ret)
                else:
                    yield self._wrapRequest(ret, generator)
                    return

            # A request with callbacks, item or None object.
            yield ret