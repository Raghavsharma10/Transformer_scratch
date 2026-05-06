def map_http_status_to_exception(http_code):
        """
        Bind a HTTP status to an HttpError.

        :param http_code: The HTTP code
        :type http_code: int

        :return The HttpError that fits to the http_code or HttpError.
        :rtype Any subclass of HttpError or HttpError
        """
        http_exceptions = HttpError.__subclasses__()
        for http_exception in http_exceptions:
            http_statuses = http_exception.HTTP_STATUSES
            if isinstance(http_statuses, int):
                http_statuses = [http_exception.HTTP_STATUSES]

            try:
                if http_code in http_statuses:
                    return http_exception
            except TypeError:  # Pass if statuses is not iterable (≈ None)
                pass

        return HttpError