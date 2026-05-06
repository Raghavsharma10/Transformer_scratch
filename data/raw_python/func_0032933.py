def result(self, wait=False):
        """
        Gets the result of the method call. If the call was successful,
        return the result, otherwise, reraise the exception.

        :param wait: Block until the result is available, or just get the result.
        :raises: RuntimeError when called and the result is not yet available.
        """
        if wait:
            self._async_resp.wait()

        if not self.finished():
            raise RuntimeError("Result is not ready yet")

        raw_response = self._async_resp.get()

        return Result(result=raw_response["result"], error=raw_response["error"],
                      id=raw_response["id"], method_call=self.request)