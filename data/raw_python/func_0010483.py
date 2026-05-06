def send_to_queue(
            self,
            args: Tuple=(),
            kwargs: Dict[str, Any]={},
            host: str=None,
            wait_result: Union[int, float]=None,
            message_ttl: Union[int, float]=None,
    ) -> Any:
        """
        Sends a message to the queue.
        A worker will run the task's function when it receives the message.

        :param args: Arguments that will be passed to task on execution.
        :param kwargs: Keyword arguments that will be passed to task
            on execution.
        :param host: Send this task to specific host. ``host`` will be
            appended to the queue name. If ``host`` is "localhost", hostname
            of the server will be appended to the queue name.
        :param wait_result:
            Wait for result from worker for ``wait_result`` seconds.
            If timeout occurs,
            :class:`~kuyruk.exceptions.ResultTimeout` is raised.
            If excecption occurs in worker,
            :class:`~kuyruk.exceptions.RemoteException` is raised.
        :param message_ttl:
            If set, message will be destroyed in queue after ``message_ttl``
            seconds.
        :return: Result from worker if ``wait_result`` is set,
            else :const:`None`.

        """
        if self.kuyruk.config.EAGER:
            # Run the task in current process
            result = self.apply(*args, **kwargs)
            return result if wait_result else None

        logger.debug("Task.send_to_queue args=%r, kwargs=%r", args, kwargs)
        queue = self._queue_for_host(host)
        description = self._get_description(args, kwargs)
        self._send_signal(signals.task_presend, args=args, kwargs=kwargs, description=description)

        body = json.dumps(description)
        msg = amqp.Message(body=body)
        if wait_result:
            # Use direct reply-to feature from RabbitMQ:
            # https://www.rabbitmq.com/direct-reply-to.html
            msg.properties['reply_to'] = 'amq.rabbitmq.reply-to'

        if message_ttl:
            msg.properties['expiration'] = str(int(message_ttl * 1000))

        with self.kuyruk.channel() as ch:
            if wait_result:
                result = Result(ch.connection)
                ch.basic_consume(queue='amq.rabbitmq.reply-to', no_ack=True, callback=result.process_message)

            ch.queue_declare(queue=queue, durable=True, auto_delete=False)
            ch.basic_publish(msg, exchange="", routing_key=queue)
            self._send_signal(signals.task_postsend, args=args, kwargs=kwargs, description=description)

            if wait_result:
                return result.wait(wait_result)