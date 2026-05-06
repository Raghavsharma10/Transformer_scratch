def aggregate_result(self, return_code, output, service_description='', specific_servers=None):
        '''
        aggregate result
        '''
        if specific_servers == None:
            specific_servers = self.servers
        else:
            specific_servers = set(self.servers).intersection(specific_servers)

        for server in specific_servers:
            if not self.servers[server]['send_errors_only'] or return_code > 0:
                self.servers[server]['results'].append({'return_code': return_code,
                           'output': output,
                           'service_description': service_description,
                           'return_status': STATUSES[return_code][0],
                           'custom_fqdn': self.servers[server]['custom_fqdn']})
                LOG.info("[email][%s][%s]: Aggregate result: %r", service_description, server, self.servers[server]['results'][-1])