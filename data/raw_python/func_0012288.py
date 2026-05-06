def publish(type, payload):
    """
        Publish a message with the specified action_type and payload over the
        event system. Useful for debugging.
    """
    async def _produce():
        # fire an action with the given values
        await producer.send(action_type=type, payload=payload)
        # notify the user that we were successful
        print("Successfully dispatched action with type {}.".format(type))

    # create a producer
    producer = ActionHandler()
    # start the producer
    producer.start()

    # get the current event loop
    loop = asyncio.get_event_loop()

    # run the production sequence
    loop.run_until_complete(_produce())

    # start the producer
    producer.stop()