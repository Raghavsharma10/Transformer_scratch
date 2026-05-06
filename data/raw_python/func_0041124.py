def get_xml_request(self):
        """ Make xml request string from stored request information.

            Returns:
                A properly formated XMl request string containing all set request fields and
                wraped in connections envelope.
        """
        def wrap_xml_content(xml_content):
            """ Wrap XML content string in the correct CPS request envelope."""
            fields = ['<?xml version="1.0" encoding="utf-8"?>\n',
                      '<cps:request xmlns:cps="www.clusterpoint.com">\n',
                      '<cps:storage>', self.connection._storage, '</cps:storage>\n']
            if self.timestamp:
                fields += []    # TODO: implement
            if self.request_id:
                fields += ['<cps:request_id>', str(self.request_id), '</cps:request_id>\n']
            if self.connection.reply_charset:
                fields += []    # TODO: implement
            if self.connection.application:
                fields += ['<cps:application>', self.connection.application, '</cps:application>\n']
            fields += ['<cps:command>', self._command, '</cps:command>\n',
                       '<cps:user>', self.connection._user, '</cps:user>\n',
                       '<cps:password>', self.connection._password, '</cps:password>\n',
                       '<cps:account>', self.connection._account, '</cps:account>\n']
            if self.timeout:
                fields += ['<cps:timeout>', str(self.timeout), '</cps:timeout>\n']
            if self.type:
                fields += ['<cps:type>', self.type, '</cps:type>\n']
            if xml_content:
                fields += ['<cps:content>\n', xml_content, '\n</cps:content>\n']
            else:
                fields += '<cps:content/>\n'
            fields += '</cps:request>\n'
            # String concat from list faster than incremental concat.
            xml_request = ''.join(fields)
            return xml_request

        xml_content = []
        if self._documents:
            xml_content += self._documents
        for key, value in self._nested_content.items():
            if value:
                xml_content += ['<{0}>'.format(key)] +\
                    ['<{0}>{1}</{0}>'.format(sub_key, sub_value) for sub_key, sub_value in value if sub_value] +\
                    ['</{0}>'.format(key)]
        for key, value in self._content.items():
            if not isinstance(value, list):
                value = [value]
            xml_content += ['<{0}>{1}</{0}>'.format(key, item) for item in value if item]
        xml_content = '\n'.join(xml_content)
        return wrap_xml_content(xml_content)