def _spawn_poll_worker_result_thread(self):
        """
        启动获取worker数据的线程
        """
        for group_id in self.group_conf:
            thread = Thread(target=self._poll_worker_result, args=(group_id,))
            thread.daemon = True
            thread.start()