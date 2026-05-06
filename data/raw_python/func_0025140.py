def grab_earliest(self, timeout: float=None) -> typing.List[DataAndMetadata.DataAndMetadata]:
        """Grab the earliest data from the buffer, blocking until one is available."""
        timeout = timeout if timeout is not None else 10.0
        with self.__buffer_lock:
            if len(self.__buffer) == 0:
                done_event = threading.Event()
                self.__done_events.append(done_event)
                self.__buffer_lock.release()
                done = done_event.wait(timeout)
                self.__buffer_lock.acquire()
                if not done:
                    raise Exception("Could not grab latest.")
            return self.__buffer.pop(0)