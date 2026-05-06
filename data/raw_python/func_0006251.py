def start(self):
        """ Starts services. """
        cert_path = os.path.join(self.work_dir, 'certificates')
        public_keys_dir = os.path.join(cert_path, 'public_keys')
        private_keys_dir = os.path.join(cert_path, 'private_keys')

        client_secret_file = os.path.join(private_keys_dir, "client.key")
        client_public, client_secret = zmq.auth.load_certificate(client_secret_file)
        server_public_file = os.path.join(public_keys_dir, "server.key")
        server_public, _ = zmq.auth.load_certificate(server_public_file)

        self.outgoing_msg_greenlet = gevent.spawn(self.outgoing_server_comms, server_public,
                                                  client_public, client_secret)
        self.outgoing_msg_greenlet.link_exception(self.on_exception)
        self.incoming_msg_greenlet = gevent.spawn(self.incoming_server_comms, server_public,
                                                  client_public, client_secret)
        self.incoming_msg_greenlet.link_exception(self.on_exception)

        logger.info('Waiting for detailed configuration from Beeswarm server.')
        gevent.joinall([self.outgoing_msg_greenlet])