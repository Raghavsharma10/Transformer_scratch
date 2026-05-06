def header_match(cls, header):
        '''
        Parse the 4-line (320-byte) library member header.
        '''
        mo = cls.header_re.match(header)
        if mo is None:
            msg = f'Expected {cls.header_re.pattern!r}, got {header!r}'
            raise ValueError(msg)
        return {
            'name': mo['name'].decode().strip(),
            'label': mo['label'].decode().strip(),
            'type': mo['type'].decode().strip(),
            'created': strptime(mo['created']),
            'modified': strptime(mo['modified']),
            'sas_version': float(mo['version']),
            'os_version': mo['os'].decode().strip(),
            'namestr_size': mo['descriptor_size'],
        }