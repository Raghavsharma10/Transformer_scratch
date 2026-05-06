def _environment_variables(**kwargs):
        # type: (Any) -> Any
        """
        Overwrite keyword arguments with environment variables

        Args:
            **kwargs: See below
            hdx_url (str): HDX url to use. Overrides hdx_site.
            hdx_site (str): HDX site to use eg. prod, test. Defaults to test.
            hdx_key (str): Your HDX key. Ignored if hdx_read_only = True.

        Returns:
            kwargs: Changed keyword arguments

        """

        hdx_key = os.getenv('HDX_KEY')
        if hdx_key is not None:
            kwargs['hdx_key'] = hdx_key
        hdx_url = os.getenv('HDX_URL')
        if hdx_url is not None:
            kwargs['hdx_url'] = hdx_url
        else:
            hdx_site = os.getenv('HDX_SITE')
            if hdx_site is not None:
                kwargs['hdx_site'] = hdx_site
        return kwargs