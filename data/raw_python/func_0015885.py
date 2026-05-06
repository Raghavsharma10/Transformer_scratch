def _parse_inspection_output(self, output):
        """Parses the machine-readable output `packer inspect` provides.

        See the inspect method for more info.
        This has been tested vs. Packer v0.7.5
        """
        parts = {'variables': [], 'builders': [], 'provisioners': []}
        for line in output.splitlines():
            line = line.split(',')
            if line[2].startswith('template'):
                del line[0:2]
                component = line[0]
                if component == 'template-variable':
                    variable = {"name": line[1], "value": line[2]}
                    parts['variables'].append(variable)
                elif component == 'template-builder':
                    builder = {"name": line[1], "type": line[2]}
                    parts['builders'].append(builder)
                elif component == 'template-provisioner':
                    provisioner = {"type": line[1]}
                    parts['provisioners'].append(provisioner)
        return parts