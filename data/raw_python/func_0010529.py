def channel(self) -> Iterator[amqp.Channel]:
        """Returns a new channel from a new connection as a context manager."""
        with self.connection() as conn:
            ch = conn.channel()
            logger.info('Opened new channel')
            with _safe_close(ch):
                yield ch