def make_access_urls(self, catalog_url, all_services, metadata=None):
        """Make fully qualified urls for the access methods enabled on the dataset.

        Parameters
        ----------
        catalog_url : str
            The top level server url
        all_services : List[SimpleService]
            list of :class:`SimpleService` objects associated with the dataset
        metadata : dict
            Metadata from the :class:`TDSCatalog`

        """
        all_service_dict = CaseInsensitiveDict({})
        for service in all_services:
            all_service_dict[service.name] = service
            if isinstance(service, CompoundService):
                for subservice in service.services:
                    all_service_dict[subservice.name] = subservice

        service_name = metadata.get('serviceName', None)

        access_urls = CaseInsensitiveDict({})
        server_url = _find_base_tds_url(catalog_url)

        # process access urls for datasets that reference top
        # level catalog services (individual or compound service
        # types).
        if service_name in all_service_dict:
            service = all_service_dict[service_name]
            if service.service_type != 'Resolver':
                # if service is a CompoundService, create access url
                # for each SimpleService
                if isinstance(service, CompoundService):
                    for subservice in service.services:
                        server_base = urljoin(server_url, subservice.base)
                        access_urls[subservice.service_type] = urljoin(server_base,
                                                                       self.url_path)
                else:
                    server_base = urljoin(server_url, service.base)
                    access_urls[service.service_type] = urljoin(server_base, self.url_path)

        # process access children of dataset elements
        for service_type in self.access_element_info:
            url_path = self.access_element_info[service_type]
            if service_type in all_service_dict:
                server_base = urljoin(server_url, all_service_dict[service_type].base)
                access_urls[service_type] = urljoin(server_base, url_path)

        self.access_urls = access_urls