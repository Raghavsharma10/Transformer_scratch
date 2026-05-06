def main():
    """Run the examples.
    """

    logging.basicConfig(level=logging.INFO)

    example_sync_client(SyncAPIClient())
    example_async_client(AsyncAPIClient())

    io_loop = ioloop.IOLoop.current()
    io_loop.start()