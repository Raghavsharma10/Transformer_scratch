def find_match_scp(self, rule):  # pylint: disable-msg=R0911,R0912
        """Handle scp commands."""

        orig_list = []
        orig_list.extend(self.original_command_list)
        binary = orig_list.pop(0)
        allowed_binaries = ['scp', '/usr/bin/scp']
        if binary not in allowed_binaries:
            self.logdebug('skipping scp processing - binary "%s" '
                          'not in approved list.\n' % binary)
            return

        filepath = orig_list.pop()
        arguments = orig_list

        if '-f' in arguments:
            if not rule.get('allow_download'):
                self.logdebug('scp denied - downloading forbidden.\n')
                return

        if '-t' in arguments:
            if not rule.get('allow_upload'):
                self.log('scp denied - uploading forbidden.\n')
                return

        if '-r' in arguments:
            if not rule.get('allow_recursion'):
                self.log('scp denied - recursive transfers forbidden.\n')
                return

        if '-p' in arguments:
            if not rule.get('allow_permissions', 'true'):
                self.log('scp denied - set/getting permissions '
                         'forbidden.\n')
                return

        if rule.get('files'):
            files = rule.get('files')
            if not isinstance(files, list):
                files = [files]
            if filepath not in files:
                self.log('scp denied - file "%s" - not in approved '
                         'list %s\n' % (filepath, files))
                return

        # Allow it!
        return {'command': self.original_command_list}