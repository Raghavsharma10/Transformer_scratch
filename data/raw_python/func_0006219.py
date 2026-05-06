def start(self):
        """ Starts services. """

        # protocol handlers
        for c in handlerbase.HandlerBase.__subclasses__():

            cap_name = c.__name__.lower()

            if cap_name in self.config['capabilities']:
                port = self.config['capabilities'][cap_name]['port']
                # carve out the options for this specific service
                options = self.config['capabilities'][cap_name]
                # capabilities are only allowed to append to the session list
                cap = c(options, self.work_dir)

                try:
                    # Convention: All capability names which end in 's' will be wrapped in ssl.
                    if cap_name.endswith('s'):
                        server = StreamServer(('0.0.0.0', port), cap.handle_session,
                                              keyfile=self.key, certfile=self.cert)
                    else:
                        server = StreamServer(('0.0.0.0', port), cap.handle_session)

                    logger.debug('Adding {0} capability with options: {1}'.format(cap_name, options))
                    self._servers.append(server)
                    server_greenlet = Greenlet(server.start())
                    self._server_greenlets.append(server_greenlet)

                except _socket.error as ex:
                    logger.error("Could not start {0} server on port {1}. Error: {2}".format(c.__name__, port, ex))
                else:
                    logger.info('Started {0} capability listening on port {1}'.format(c.__name__, port))

        stop_if_not_write_workdir(self.work_dir)
        logger.info("Honeypot running.")

        gevent.joinall(self._server_greenlets)