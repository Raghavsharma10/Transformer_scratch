def parse(self, file_obj):
        """
        Read an OpenSSH config from the given file object.

        @param file_obj: a file-like object to read the config file from
        @type file_obj: file
        """
        configs = [self._config[0]]
        for line in file_obj:
            line = line.rstrip('\n').lstrip()
            if (line == '') or (line[0] == '#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip().lower()
            else:
                # find first whitespace, and split there
                i = 0
                while (i < len(line)) and not line[i].isspace():
                    i += 1
                if i == len(line):
                    raise Exception('Unparsable line: %r' % line)
                key = line[:i].lower()
                value = line[i:].lstrip()

            if key == 'host':
                del configs[:]
                # the value may be multiple hosts, space-delimited
                for host in value.split():
                    # do we have a pre-existing host config to append to?
                    matches = [c for c in self._config if c['host'] == host]
                    if len(matches) > 0:
                        configs.append(matches[0])
                    else:
                        config = { 'host': host }
                        self._config.append(config)
                        configs.append(config)
            else:
                for config in configs:
                    config[key] = value