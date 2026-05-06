def check_client_ip(self, rule):
        """If a client IP is specified, verify it is permitted."""

        if not rule.get('from'):
            self.logdebug('no "from" requirement.\n')
            return True

        allow_from = rule.get('from')
        if not isinstance(allow_from, list):
            allow_from = [allow_from]
        client_ip = self.get_client_ip()

        if client_ip in allow_from:
            self.logdebug('client_ip %s in %s\n' % (client_ip, allow_from))
            return True
        else:
            self.logdebug('client_ip %s not in %s' % (client_ip, allow_from))
            return False