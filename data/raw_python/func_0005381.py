def search_datasets(
        self,
        license=None,
        format=None,
        query=None,
        featured=None,
        owner=None,
        organization=None,
        badge=None,
        reuses=None,
        page_size=20,
        x_fields=None,
    ):
        """Search datasets within uData portal."""
        # handling request parameters
        payload = {"badge": badge, "size": page_size, "X-Fields": x_fields}

        # search request
        # head = {"X-API-KEY": self.api_key}
        search_url = "{}/datasets".format(
            self.base_url,
            # org_id,
            # page_size
        )

        search_req = requests.get(
            search_url,
            # headers=head,
            params=payload,
        )

        # serializing result into dict and storing resources in variables
        logger.debug(search_req.url)
        return search_req.json()