def add_access_element_info(self, access_element):
        """Create an access method from a catalog element."""
        service_name = access_element.attrib['serviceName']
        url_path = access_element.attrib['urlPath']
        self.access_element_info[service_name] = url_path