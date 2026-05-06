def _parse_header(self, data):
        """Parse header (xheader or yheader)

        :param data: data to be parsed
        :type data: str
        :return: list with header's data
        :rtype: list
        """
        return_list = []

        headers = data.split(':')

        for header in headers:
            header = re.split(' IN ', header, flags=re.I) # ignore case
            xheader = {'name': header[0].strip()}
            if len(header) > 1:
                xheader['units'] = header[1].strip()
            return_list.append(xheader)

        return return_list