def main(self):
        """The main function containing the loop for communication and process management.

        This function is the heart of the daemon.
        It is responsible for:
        - Client communication
        - Executing commands from clients
        - Update the status of processes by polling the ProcessHandler.
        - Logging
        - Cleanup on exit

        """
        try:
            while self.running:
                # Trigger the processing of finished processes by the ProcessHandler.
                # If there are finished processes we write the log to keep it up to date.
                if self.process_handler.check_finished():
                    self.logger.write(self.queue)

                if self.reset and self.process_handler.all_finished():
                    # Rotate log and reset queue
                    self.logger.rotate(self.queue)
                    self.queue.reset()
                    self.reset = False

                # Check if the ProcessHandler has any free slots to spawn a new process
                if not self.paused and not self.reset and self.running:
                    self.process_handler.check_for_new()

                # This is the communication section of the daemon.
                # 1. Receive message from the client
                # 2. Check payload and call respective function with payload as parameter.
                # 3. Execute logic
                # 4. Return payload with response to client

                # Create list for waitable objects
                readable, writable, failed = select.select(self.read_list, [], [], 1)
                for waiting_socket in readable:
                    if waiting_socket is self.socket:
                        # Listening for clients to connect.
                        # Client sockets are added to readlist to be processed.
                        try:
                            client_socket, client_address = self.socket.accept()
                            self.read_list.append(client_socket)
                        except Exception:
                            self.logger.warning('Daemon rejected client')
                    else:
                        # Trying to receive instruction from client socket
                        try:
                            instruction = waiting_socket.recv(1048576)
                        except (EOFError, OSError):
                            self.logger.warning('Client died while sending message, dropping received data.')
                            # Remove client socket
                            self.read_list.remove(waiting_socket)
                            waiting_socket.close()
                            instruction = None

                        # Check for valid instruction
                        if instruction is not None:
                            # Check if received data can be unpickled.
                            try:
                                payload = pickle.loads(instruction)
                            except EOFError:
                                # Instruction is ignored if it can't be unpickled
                                self.logger.error('Received message is incomplete, dropping received data.')
                                self.read_list.remove(waiting_socket)
                                waiting_socket.close()
                                # Set invalid payload
                                payload = {'mode': ''}

                            functions = {
                                'add': self.add,
                                'remove': self.remove,
                                'edit': self.edit_command,
                                'switch': self.switch,
                                'send': self.pipe_to_process,
                                'status': self.send_status,
                                'start': self.start,
                                'pause': self.pause,
                                'stash': self.stash,
                                'enqueue': self.enqueue,
                                'restart': self.restart,
                                'kill': self.kill_process,
                                'reset': self.reset_everything,
                                'clear': self.clear,
                                'config': self.set_config,
                                'STOPDAEMON': self.stop_daemon,
                            }

                            if payload['mode'] in functions.keys():
                                self.logger.debug('Payload received:')
                                self.logger.debug(payload)
                                response = functions[payload['mode']](payload)

                                self.logger.debug('Sending payload:')
                                self.logger.debug(response)
                                try:
                                    self.respond_client(response, waiting_socket)
                                except (BrokenPipeError):
                                    self.logger.warning('Client disconnected during message dispatching. Function successfully executed anyway.')
                                    # Remove client socket
                                    self.read_list.remove(waiting_socket)
                                    waiting_socket.close()
                                    instruction = None
                            else:
                                self.respond_client({'message': 'Unknown Command',
                                                    'status': 'error'}, waiting_socket)
        except Exception:
            self.logger.exception()

        # Wait for killed or stopped processes to finish (cleanup)
        self.process_handler.wait_for_finish()
        # Close socket, clean everything up and exit
        self.socket.close()
        cleanup(self.config_dir)
        sys.exit(0)