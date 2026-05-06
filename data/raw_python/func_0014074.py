def _get_diff(self, cp_file):
        """Get a diff between running config and a proposed file."""
        diff = []
        self._create_sot_file()
        diff_out = self.device.show(
            'show diff rollback-patch file {0} file {1}'.format(
                'sot_file', self.replace_file.split('/')[-1]), raw_text=True)
        try:
            diff_out = diff_out.split(
                '#Generating Rollback Patch')[1].replace(
                    'Rollback Patch is Empty', '').strip()
            for line in diff_out.splitlines():
                if line:
                    if line[0].strip() != '!':
                        diff.append(line.rstrip(' '))
        except (AttributeError, KeyError):
            raise ReplaceConfigException(
                'Could not calculate diff. It\'s possible the given file doesn\'t exist.')
        return '\n'.join(diff)