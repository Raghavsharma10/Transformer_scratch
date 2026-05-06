def begin_stream_loop(stream, poll_interval):
    """Start and maintain the streaming connection..."""
    while should_continue():
        try:
            stream.start_polling(poll_interval)
        except Exception as e:
            # Infinite restart
            logger.error("Exception while polling. Restarting in 1 second.", exc_info=True)
            time.sleep(1)