def launch_plugin(self):
        '''
        launch nagios_plugin command
        '''
        # nagios_plugins probes
        for plugin in self.plugins:
            # Construct the nagios_plugin command
            command = ('%s%s' % (self.plugins[plugin]['path'], self.plugins[plugin]['command']))

            try:
                nagios_plugin = subprocess.Popen(command,
                                                 shell=True,
                                                 stdout=subprocess.PIPE,
                                                 stderr=subprocess.PIPE)
            except OSError:
                LOG.error("[nagios_plugins]: '%s' executable is missing",
                          command)
            else:
                output = nagios_plugin.communicate()[0].strip()
                return_code = nagios_plugin.returncode
                if return_code >= len(STATUSES):
                    LOG.error("[nagios_plugins]: '%s' executable has an issue, return code: %s",
                              command, return_code)
                else:
                    LOG.log(STATUSES[return_code][1],
                                "[nagios_plugins][%s] (%s status): %s",
                                plugin,
                                STATUSES[return_code][0],
                                output)
                    yield {'return_code': int(return_code),
                           'output': str(output),
                           'time_stamp': int(time.time()),
                           'service_description': plugin,
                           'specific_servers': self.plugins[plugin]['servers']}