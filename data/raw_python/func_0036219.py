def _parse_info(line):
        """
        The output can be:
        - [LaCrosseITPlusReader.10.1s (RFM12B f:0 r:17241)]
        - [LaCrosseITPlusReader.10.1s (RFM12B f:0 t:10~3)]
        """
        re_info = re.compile(
            r'\[(?P<name>\w+).(?P<ver>.*) ' +
            r'\((?P<rfm1name>\w+) (\w+):(?P<rfm1freq>\d+) ' +
            r'(?P<rfm1mode>.*)\)\]')

        info = {
            'name': None,
            'version': None,
            'rfm1name': None,
            'rfm1frequency': None,
            'rfm1datarate': None,
            'rfm1toggleinterval': None,
            'rfm1togglemask': None,
        }
        match = re_info.match(line)
        if match:
            info['name'] = match.group('name')
            info['version'] = match.group('ver')
            info['rfm1name'] = match.group('rfm1name')
            info['rfm1frequency'] = match.group('rfm1freq')
            values = match.group('rfm1mode').split(':')
            if values[0] == 'r':
                info['rfm1datarate'] = values[1]
            elif values[0] == 't':
                toggle = values[1].split('~')
                info['rfm1toggleinterval'] = toggle[0]
                info['rfm1togglemask'] = toggle[1]

        return info