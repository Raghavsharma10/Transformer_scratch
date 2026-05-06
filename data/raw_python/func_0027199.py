def run(self):
        """
        执行任务
        """
        while not self._stoped:
            self._tx_event.wait()
            self._tx_event.clear()
            try:
                func = self._tx_queue.get_nowait()
                if isinstance(func, str):
                    self._stoped = True
                    self._rx_queue.put('closed')
                    self.notice()
                    break
            except Empty:
                # pragma: no cover
                continue
            try:
                result = func()
                self._rx_queue.put(result)
            except Exception as e:
                self._rx_queue.put(e)
            self.notice()
        else:
            # pragma: no cover
            pass