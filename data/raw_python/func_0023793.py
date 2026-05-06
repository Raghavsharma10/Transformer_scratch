def _executor(self):
        '''
            Creating an ExecutorPool is a costly operation. Executor needs to be instantiated only once.
        '''
        if self.EXECUTE_PARALLEL is False:
            executor_path = "batch_requests.concurrent.executor.SequentialExecutor"
            executor_class = import_class(executor_path)
            return executor_class()
        else:
            executor_path = self.CONCURRENT_EXECUTOR
            executor_class = import_class(executor_path)
            return executor_class(self.NUM_WORKERS)