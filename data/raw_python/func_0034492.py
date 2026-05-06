def check_keyname(self, rule):
        """If a key name is specified, verify it is permitted."""

        keynames = rule.get('keynames')
        if not keynames:
            self.logdebug('no keynames requirement.\n')
            return True
        if not isinstance(keynames, list):
            keynames = [keynames]

        if self.keyname in keynames:
            self.logdebug('keyname "%s" matches rule.\n' % self.keyname)
            return True
        else:
            self.logdebug('keyname "%s" does not match rule.\n' % self.keyname)
            return False