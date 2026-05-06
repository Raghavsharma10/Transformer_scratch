def parse(self, response):
        """
        Parse the login xml response
        :param response: the login response from the RETS server
        :return: None
        """
        self.headers = response.headers

        if 'xml' in self.headers.get('Content-Type'):
            # Got an XML response, likely an error code.
            xml = xmltodict.parse(response.text)
            self.analyze_reply_code(xml_response_dict=xml)

        lines = response.text.split('\r\n')
        if len(lines) < 3:
            lines = response.text.split('\n')

        for line in lines:
            line = line.strip()

            name, value = self.read_line(line)
            if name:
                if name in self.valid_transactions or re.match(pattern='/^X\-/', string=name):
                    self.capabilities[name] = value
                else:
                    self.details[name] = value