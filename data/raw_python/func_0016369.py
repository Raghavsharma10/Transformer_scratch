def monitor(self):
        """Flushes the queue periodically."""
        while self.monitor_running.is_set():
            if time.time() - self.last_flush > self.batch_time:
                if not self.queue.empty():
                    logger.info("Queue Flush: time without flush exceeded")
                    self.flush_queue()
            time.sleep(self.batch_time)