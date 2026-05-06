def _wait_for_queue_space(self, timeout=DEFAULT_WAIT_TIME):
        """
        EXPECT THE self.lock TO BE HAD, WAITS FOR self.queue TO HAVE A LITTLE SPACE
        """
        wait_time = 5

        (DEBUG and len(self.queue) > 1 * 1000 * 1000) and Log.warning("Queue {{name}} has over a million items")

        now = time()
        if timeout != None:
            time_to_stop_waiting = now + timeout
        else:
            time_to_stop_waiting = now + DEFAULT_WAIT_TIME

        if self.next_warning < now:
            self.next_warning = now + wait_time

        while not self.closed and len(self.queue) >= self.max:
            if now > time_to_stop_waiting:
                Log.error(THREAD_TIMEOUT)

            if self.silent:
                self.lock.wait(Till(till=time_to_stop_waiting))
            else:
                self.lock.wait(Till(seconds=wait_time))
                if len(self.queue) >= self.max:
                    now = time()
                    if self.next_warning < now:
                        self.next_warning = now + wait_time
                        Log.alert(
                            "Queue by name of {{name|quote}} is full with ({{num}} items), thread(s) have been waiting {{wait_time}} sec",
                            name=self.name,
                            num=len(self.queue),
                            wait_time=wait_time
                        )