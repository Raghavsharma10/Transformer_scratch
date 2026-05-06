def _assert_command_dict(self, struct, name, path=None, extra_info=None):
        """Checks whether struct is a command dict (e.g. it's a dict and has 1 key-value pair."""
        self._assert_dict(struct, name, path, extra_info)
        if len(struct) != 1:
            err = [self._format_error_path(path + [name])]
            err.append('Commands of run, dependencies, and argument sections must be mapping with '
                       'exactly 1 key-value pair, got {0}: {1}'.format(len(struct), struct))
            if extra_info:
                err.append(extra_info)
            raise exceptions.YamlSyntaxError('\n'.join(err))