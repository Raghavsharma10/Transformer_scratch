def search(self, entity_type, property_name, search_string, start_index=0, max_results=99999):
        """Performs a user search using the Crowd search API.

        https://developer.atlassian.com/display/CROWDDEV/Crowd+REST+Resources#CrowdRESTResources-SearchResource

        Args:
            entity_type: 'user' or 'group'
            property_name: eg. 'email', 'name'
            search_string: the string to search for.
            start_index: starting index of the results (default: 0)
            max_results: maximum number of results returned (default: 99999)

        Returns:
            json results:
                Returns search results.
        """

        params = {
            "entity-type": entity_type,
            "expand": entity_type,
            "property-search-restriction": {
                "property": {"name": property_name, "type": "STRING"},
                "match-mode": "CONTAINS",
                "value": search_string,
            }
        }

        params = {
            'entity-type': entity_type,
            'expand': entity_type,
            'start-index': start_index,
            'max-results': max_results
        }
        # Construct XML payload of the form:
        # <property-search-restriction>
        #   <property>
        #     <name>email</name>
        #     <type>STRING</type>
        #   </property>
        #   <match-mode>EXACTLY_MATCHES</match-mode>
        #   <value>bob@example.net</value>
        # </property-search-restriction>

        root = etree.Element('property-search-restriction')

        property_ = etree.Element('property')
        prop_name = etree.Element('name')
        prop_name.text = property_name
        property_.append(prop_name)
        prop_type = etree.Element('type')
        prop_type.text = 'STRING'
        property_.append(prop_type)
        root.append(property_)

        match_mode = etree.Element('match-mode')
        match_mode.text = 'CONTAINS'
        root.append(match_mode)

        value = etree.Element('value')
        value.text = search_string
        root.append(value)

        # Construct the XML payload expected by search API
        payload = '<?xml version="1.0" encoding="UTF-8"?>\n' + etree.tostring(root).decode('utf-8')

        # We're sending XML but would like a JSON response
        session = self._build_session(content_type='xml')
        session.headers.update({'Accept': 'application/json'})
        response = session.post(self.rest_url + "/search", params=params, data=payload, timeout=self.timeout)

        if not response.ok:
            return None

        return response.json()