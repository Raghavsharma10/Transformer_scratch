def get_info(self):
        """Get current configuration info from 'v' command."""
        re_info = re.compile(r'\[.*\]')

        self._write_cmd('v')
        while True:
            line = self._serial.readline()
            try:
                line = line.encode().decode('utf-8')
            except AttributeError:
                line = line.decode('utf-8')

            match = re_info.match(line)
            if match:
                return self._parse_info(line)