def message_proxy(self, work_dir):
        """
        drone_data_inboud   is for data comming from drones
        drone_data_outbound is for commands to the drones, topic must either be a drone ID or all for sending
                            a broadcast message to all drones
        """
        public_keys_dir = os.path.join(work_dir, 'certificates', 'public_keys')
        secret_keys_dir = os.path.join(work_dir, 'certificates', 'private_keys')

        # start and configure auth worker
        auth = IOLoopAuthenticator()
        auth.start()
        auth.allow('127.0.0.1')
        auth.configure_curve(domain='*', location=public_keys_dir)

        # external interfaces for communicating with drones
        server_secret_file = os.path.join(secret_keys_dir, 'beeswarm_server.pri')
        server_public, server_secret = load_certificate(server_secret_file)
        drone_data_inbound = beeswarm.shared.zmq_context.socket(zmq.PULL)
        drone_data_inbound.curve_secretkey = server_secret
        drone_data_inbound.curve_publickey = server_public
        drone_data_inbound.curve_server = True
        drone_data_inbound.bind('tcp://*:{0}'.format(self.config['network']['zmq_port']))

        drone_data_outbound = beeswarm.shared.zmq_context.socket(zmq.PUB)
        drone_data_outbound.curve_secretkey = server_secret
        drone_data_outbound.curve_publickey = server_public
        drone_data_outbound.curve_server = True
        drone_data_outbound.bind('tcp://*:{0}'.format(self.config['network']['zmq_command_port']))

        # internal interfaces
        # all inbound session data from drones will be replayed on this socket
        drone_data_socket = beeswarm.shared.zmq_context.socket(zmq.PUB)
        drone_data_socket.bind(SocketNames.DRONE_DATA.value)

        # all commands received on this will be published on the external interface
        drone_command_socket = beeswarm.shared.zmq_context.socket(zmq.PULL)
        drone_command_socket.bind(SocketNames.DRONE_COMMANDS.value)

        poller = zmq.Poller()
        poller.register(drone_data_inbound, zmq.POLLIN)
        poller.register(drone_command_socket, zmq.POLLIN)
        while True:
            # .recv() gives no context switch - why not? using poller with timeout instead
            socks = dict(poller.poll(100))
            gevent.sleep()

            if drone_command_socket in socks and socks[drone_command_socket] == zmq.POLLIN:
                data = drone_command_socket.recv()
                drone_id, _ = data.split(' ', 1)
                logger.debug("Sending drone command to: {0}".format(drone_id))
                # pub socket takes care of filtering
                drone_data_outbound.send(data)
            elif drone_data_inbound in socks and socks[drone_data_inbound] == zmq.POLLIN:
                raw_msg = drone_data_inbound.recv()
                split_data = raw_msg.split(' ', 2)
                if len(split_data) == 3:
                    topic, drone_id, data = split_data
                else:
                    data = None
                    topic, drone_id, = split_data
                logger.debug("Received {0} message from {1}.".format(topic, drone_id))
                # relay message on internal socket
                drone_data_socket.send(raw_msg)