def parse_image_response(self, response):
        """
        Parse multiple objects from the RETS feed. A lot of string methods are used to handle the response before
        encoding it back into bytes for the object.
        :param response: The response from the feed
        :return: list of SingleObjectParser
        """
        if 'xml' in response.headers.get('Content-Type'):
            # Got an XML response, likely an error code.
            xml = xmltodict.parse(response.text)
            self.analyze_reply_code(xml_response_dict=xml)

        multi_parts = self._get_multiparts(response)
        parsed = []
        # go through each part of the multipart message
        for part in multi_parts:
            clean_part = part.strip('\r\n\r\n')
            if '\r\n\r\n' in clean_part:
                header, body = clean_part.split('\r\n\r\n', 1)
            else:
                header = clean_part
                body = None
            part_header_dict = {k.strip(): v.strip() for k, v in (h.split(':', 1) for h in header.split('\r\n'))}

            # Some multipart requests respond with a text/XML part stating an error
            if 'xml' in part_header_dict.get('Content-Type'):
                # Got an XML response, likely an error code.
                # Some rets servers give characters after the closing brace.
                body = body[:body.index('/>') + 2]  if '/>' in body else body
                xml = xmltodict.parse(body)
                try:
                    self.analyze_reply_code(xml_response_dict=xml)
                except RETSException as e:
                    if e.reply_code == '20403':
                        # The requested object_id was not found.
                        continue
                    raise e

            if body:
                obj = self._response_object_from_header(
                    obj_head_dict=part_header_dict,
                    content=body.encode('latin-1') if six.PY3 else body)
            else:
                obj = self._response_object_from_header(obj_head_dict=part_header_dict)
            parsed.append(obj)
        return parsed