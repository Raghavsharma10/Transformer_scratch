def _extract_query(self, redirect_url):
        """Extract query parameters from a url.
        Parameters
            redirect_url (str)
                The full URL that the Lyft server redirected to after
                the user authorized your app.
        Returns
            (dict)
                A dictionary of query parameters.
        """
        qs = urlparse(redirect_url)

        # redirect_urls return data after query identifier (?)
        qs = qs.query

        query_params = parse_qs(qs)
        query_params = {qp: query_params[qp][0] for qp in query_params}

        return query_params