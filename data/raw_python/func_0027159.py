def _did_receive_response(self, connection):
        """ Receive a response from the connection """

        if connection.has_timeouted:
            bambou_logger.info("NURESTConnection has timeout.")
            return

        has_callbacks = connection.has_callbacks()
        should_post = not has_callbacks

        if connection.handle_response_for_connection(should_post=should_post) and has_callbacks:
            callback = connection.callbacks['local']
            callback(connection)