def urlopen(self, url, **kwargs):
        """GET a file-like object for a URL using HTTP.

        This is a thin wrapper around :meth:`requests.Session.get` that returns a file-like
        object wrapped around the resulting content.

        Parameters
        ----------
        url : str
            The URL to request

        kwargs : arbitrary keyword arguments
            Additional keyword arguments to pass to :meth:`requests.Session.get`.

        Returns
        -------
        fobj : file-like object
            A file-like interface to the content in the response

        See Also
        --------
        :meth:`requests.Session.get`

        """
        return BytesIO(self.create_session().get(url, **kwargs).content)