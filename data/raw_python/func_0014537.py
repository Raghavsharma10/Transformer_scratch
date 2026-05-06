def page_through(page_size, function, *args, **kwargs):
        """Return an iterator over all pages of a REST operation.

        :param page_size: Number of elements to retrieve per call.
        :param function: FlashArray function that accepts limit as an argument.
        :param \*args: Positional arguments to be passed to function.
        :param \*\*kwargs: Keyword arguments to be passed to function.

        :returns: An iterator of tuples containing a page of results for the
                  function(\*args, \*\*kwargs) and None, or None and a PureError
                  if a call to retrieve a page fails.
        :rtype: iterator

        .. note::

            Requires use of REST API 1.7 or later.

            Only works with functions that accept limit as an argument.

            Iterator will retrieve page_size elements per call

            Iterator will yield None and an error if a call fails. The next
            call will repeat the same call, unless the caller sends in an
            alternate page token.

        """

        kwargs["limit"] = page_size

        def get_page(token):
            page_kwargs = kwargs.copy()
            if token:
                page_kwargs["token"] = token
            return function(*args, **page_kwargs)

        def page_generator():
            token = None
            while True:
                try:
                    response = get_page(token)
                    token = response.headers.get("x-next-token")
                except PureError as err:
                    yield None, err
                else:
                    if response:
                        sent_token = yield response, None
                        if sent_token is not None:
                            token = sent_token
                    else:
                        return

        return page_generator()