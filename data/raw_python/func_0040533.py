def write_response(
        self, status_code: Union[int, constants.HttpStatusCode], *,
        headers: Optional[_HeaderType]=None
            ) -> "writers.HttpResponseWriter":
        """
        Write a response to the client.
        """
        self._writer = self.__delegate.write_response(
            constants.HttpStatusCode(status_code),
            headers=headers)

        return self._writer