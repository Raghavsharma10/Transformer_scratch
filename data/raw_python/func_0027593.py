def _sub_process_main(started_event: Event,
                      channel_name: str,
                      connection: Connection,
                      consumer_configuration: BrightsideConsumerConfiguration,
                      consumer_factory: Callable[[Connection, BrightsideConsumerConfiguration, logging.Logger], BrightsideConsumer],
                      command_processor_factory: Callable[[str], CommandProcessor],
                      mapper_func: Callable[[BrightsideMessage], Request]) -> None:
    """
    This is the main method for the sub=process, everything we need to create the message pump and
    channel it needs to be passed in as parameters that can be pickled as when we run they will be serialized
    into this process. The data should be value types, not reference types as we will receive a copy of the original.
    Inter-process communication is signalled by the event - to indicate startup - and the pipeline to facilitate a
    sentinel or stop message
    :param started_event: Used by the sub-process to signal that it is ready
    :param channel_name: The name we want to give the channel to the broker for identification
    :param connection: The 'broker' connection
    :param consumer_configuration: How to configure our consumer of messages from the channel
    :param consumer_factory: Callback to create the consumer. User code as we don't know what consumer library they
        want to use. Arame? Something else?
    :param command_processor_factory: Callback to  register subscribers, policies, and task queues then build command
        processor. User code that provides us with their requests and handlers
    :param mapper_func: We need to map between messages on the wire and our handlers
    :return:
    """

    logger = logging.getLogger(__name__)
    consumer = consumer_factory(connection, consumer_configuration, logger)
    channel = Channel(name=channel_name, consumer=consumer, pipeline=consumer_configuration.pipeline)

    # TODO: Fix defaults that need passed in config values
    command_processor = command_processor_factory(channel_name)
    message_pump = MessagePump(command_processor=command_processor, channel=channel, mapper_func=mapper_func,
                               timeout=500, unacceptable_message_limit=None, requeue_count=None)

    logger.debug("Starting the message pump for %s", channel_name)
    message_pump.run(started_event)