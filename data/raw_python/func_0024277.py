def check_mq_connection(self):
        """
        RabbitMQ checks the connection
        It displays on the screen whether or not you have a connection.
        """
        import pika
        from zengine.client_queue import BLOCKING_MQ_PARAMS
        from pika.exceptions import ProbableAuthenticationError, ConnectionClosed

        try:
            connection = pika.BlockingConnection(BLOCKING_MQ_PARAMS)
            channel = connection.channel()
            if channel.is_open:
                print(__(u"{0}RabbitMQ is working{1}").format(CheckList.OKGREEN, CheckList.ENDC))
            elif self.channel.is_closed or self.channel.is_closing:
                print(__(u"{0}RabbitMQ is not working!{1}").format(CheckList.FAIL, CheckList.ENDC))
        except ConnectionClosed as e:
            print(__(u"{0}RabbitMQ is not working!{1}").format(CheckList.FAIL, CheckList.ENDC), e)
        except ProbableAuthenticationError as e:
            print(__(u"{0}RabbitMQ username and password wrong{1}").format(CheckList.FAIL,
                                                                           CheckList.ENDC))