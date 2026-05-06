def send(self):
        """ Send an XML string version of content through the connection.

        Returns:
            Response object.
        """
        xml_request = self.get_xml_request()
        if(self.connection._debug == 1):
            print(xml_request)
        Debug.warn('-' * 25)
        Debug.warn(self._command)
        Debug.dump("doc: \n", self._documents)
        Debug.dump("cont: \n", self._content)
        Debug.dump("nest cont \n", self._nested_content)
        Debug.dump("Request: \n", xml_request)


        response = _handle_response(self.connection._send_request(xml_request),
                                         self._command, self.connection.document_id_xpath)
        # TODO: jāpabeidz debugs 
        # if(self.connection._debug == 1):
        #     # print(response)
        #     print(format(ET.tostring(response)))
        return response