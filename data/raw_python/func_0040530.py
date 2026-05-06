def write_response(
        self, status_code: Union[
            int, constants.HttpStatusCode
        ]=constants.HttpStatusCode.BAD_REQUEST, *,
        headers: Optional[_HeaderType]=None
            ) -> "writers.HttpResponseWriter":
        """
        When this exception is raised on the server side, this method is used
        to send a error response instead of
        :method:`BaseHttpStreamReader.write_response()`.
        """
        return self._delegate.write_response(
            constants.HttpStatusCode(status_code),
            headers=headers)