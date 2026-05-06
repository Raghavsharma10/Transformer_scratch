def _set_worker_thread_level(self):
        """Sets logging level of the background logging thread to DEBUG or INFO
        """
        bthread_logger = logging.getLogger(
            'google.cloud.logging.handlers.transports.background_thread')
        if self.debug_thread_worker:
            bthread_logger.setLevel(logging.DEBUG)
        else:
            bthread_logger.setLevel(logging.INFO)