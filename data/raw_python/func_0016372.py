def close(self):
        """Flushes the queue and waits for the executor to finish."""
        logger.info('Closing producer')
        self.flush_queue()
        self.monitor_running.clear()
        self.pool.shutdown()
        logger.info('Producer closed')