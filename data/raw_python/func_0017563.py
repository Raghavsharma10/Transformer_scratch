def parse(self, response, metadata_type):
        """
        Parses RETS metadata using the STANDARD-XML format
        :param response: requests Response object
        :param metadata_type: string
        :return parsed: list
        """
        xml = xmltodict.parse(response.text)
        self.analyze_reply_code(xml_response_dict=xml)
        base = xml.get('RETS', {}).get('METADATA', {}).get(metadata_type, {})

        if metadata_type == 'METADATA-SYSTEM':
            syst = base.get('System', base.get('SYSTEM'))
            if not syst:
                raise ParseError("Could not get the System key from a METADATA-SYSTEM request.")

            system_obj = {}
            if syst.get('SystemID'):
                system_obj['system_id'] = str(syst['SystemID'])
            if syst.get('SystemDescription'):
                system_obj['system_description'] = str(syst['SystemDescription'])
            if syst.get('Comments'):
                system_obj['comments'] = syst['Comments']
            if base.get('@Version'):
                system_obj['version'] = base['@Version']
            return [system_obj]

        elif metadata_type == 'METADATA-CLASS':
            key = 'class'
        elif metadata_type == 'METADATA-RESOURCE':
            key = 'resource'
        elif metadata_type == 'METADATA-LOOKUP_TYPE':
            key = 'lookuptype'
        elif metadata_type == 'METADATA-OBJECT':
            key = 'object'
        elif metadata_type == 'METADATA-TABLE':
            key = 'field'
        else:
            msg = "Got an unknown metadata type of {0!s}".format(metadata_type)
            raise ParseError(msg)

        # Get the version with the right capitalization from the dictionary
        key_cap = None
        for k in base.keys():
            if k.lower() == key:
                key_cap = k

        if not key_cap:
            msg = 'Could not find {0!s} in the response XML'.format(key)
            raise ParseError(msg)

        if isinstance(base[key_cap], list):
            return base[key_cap]
        else:
            return [base[key_cap]]