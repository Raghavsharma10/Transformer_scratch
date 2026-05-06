def run_forever(self, start_at='once'):
        """
        Starts the scheduling engine
        
        @param start_at: 'once' -> start immediately
                         'next_minute' -> start at the first second of the next minutes
                         'next_hour' -> start 00:00 (min) next hour
                         'tomorrow' -> start at 0h tomorrow
        """
        if start_at not in ('once', 'next_minute', 'next_hour', 'tomorrow'):
            raise ValueError("start_at parameter must be one of these values: 'once', 'next_minute', 'next_hour', 'tomorrow'")
        if start_at != 'once':
            wait_until(start_at)
        try:
            task_pool = self.run_tasks()
            while self.running:
                gevent.sleep(seconds=1)
            task_pool.join(timeout=30)
            task_pool.kill()
        except KeyboardInterrupt:
            # https://github.com/surfly/gevent/issues/85
            task_pool.closed = True
            task_pool.kill()
            logging.getLogger(self.logger_name).info('Time scheduler quits')