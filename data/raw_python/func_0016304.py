def requestTimedOut(self, reply):
        """Trap the timeout. In Async mode requestTimedOut is called after replyFinished"""
        # adapt http_call_result basing on receiving qgs timer timout signal
        self.exception_class = RequestsExceptionTimeout
        self.http_call_result.exception = RequestsExceptionTimeout("Timeout error")