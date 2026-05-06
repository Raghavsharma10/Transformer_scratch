def parse_image_response(self, response):
        """
        Parse a single object from the RETS feed
        :param response: The response from the RETS server
        :return: Object
        """
        if 'xml' in response.headers.get('Content-Type'):
            # Got an XML response, likely an error code.
            xml = xmltodict.parse(response.text)
            self.analyze_reply_code(xml_response_dict=xml)

        obj = self._response_object_from_header(
            obj_head_dict=response.headers,
            content=response.content)
        return obj