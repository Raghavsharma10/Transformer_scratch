def _heartbeat(self):
        """
        **Purpose**: Method to be executed in the heartbeat thread. This method sends a 'request' to the
        heartbeat-req queue. It expects a 'response' message from the 'heartbeart-res' queue within 10 seconds. This
        message should contain the same correlation id. If no message if received in 10 seconds, the tmgr is assumed
        dead. The end_manager() is called to cleanly terminate tmgr process and the heartbeat thread is also 
        terminated.

        **Details**: The AppManager can re-invoke both if the execution is still not complete.
        """

        try:

            self._prof.prof('heartbeat thread started', uid=self._uid)

            mq_connection = pika.BlockingConnection(pika.ConnectionParameters(host=self._mq_hostname, port=self._port))
            mq_channel = mq_connection.channel()

            response = True
            while (response and (not self._hb_terminate.is_set())):
                response = False
                corr_id = str(uuid.uuid4())

                # Heartbeat request signal sent to task manager via rpc-queue
                mq_channel.basic_publish(exchange='',
                                         routing_key=self._hb_request_q,
                                         properties=pika.BasicProperties(
                                             reply_to=self._hb_response_q,
                                             correlation_id=corr_id),
                                         body='request')
                self._logger.info('Sent heartbeat request')

                # mq_connection.close()

                # Sleep for hb_interval and then check if tmgr responded
                mq_connection.sleep(self._hb_interval)

                # mq_connection = pika.BlockingConnection(
                #     pika.ConnectionParameters(host=self._mq_hostname, port=self._port))
                # mq_channel = mq_connection.channel()

                method_frame, props, body = mq_channel.basic_get(queue=self._hb_response_q)

                if body:
                    if corr_id == props.correlation_id:
                        self._logger.info('Received heartbeat response')
                        response = True

                        mq_channel.basic_ack(delivery_tag=method_frame.delivery_tag)

                # Appease pika cos it thinks the connection is dead
                # mq_connection.close()

        except KeyboardInterrupt:
            self._logger.exception('Execution interrupted by user (you probably hit Ctrl+C), ' +
                               'trying to cancel tmgr process gracefully...')
            raise KeyboardInterrupt

        except Exception as ex:
            self._logger.exception('Heartbeat failed with error: %s' % ex)
            raise

        finally:

            try:
                mq_connection.close()
            except:
                self._logger.warning('mq_connection not created')

            self._prof.prof('terminating heartbeat thread', uid=self._uid)