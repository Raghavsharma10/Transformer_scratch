def match(header):
        '''
        Parse the 3-line (240-byte) header of a SAS XPORT file.
        '''
        mo = Library.header_re.match(header)
        if mo is None:
            raise ValueError(f'Not a SAS Version 5 or 6 XPORT file')
        return {
            'created': strptime(mo['created']),
            'modified': strptime(mo['modified']),
            'sas_version': float(mo['version']),
            'os_version': mo['os'].decode().strip(),
        }