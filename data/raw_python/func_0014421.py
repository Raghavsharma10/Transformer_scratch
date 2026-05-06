def get_multi_q(self, sentinel='STOP'):
        '''
        This helps indexq operate in multiprocessing environment without each process having to have it's own IndexQ. It also is a handy way to deal with thread / process safety.

        This method will create and return a JoinableQueue object. Additionally, it will kick off a back end process that will monitor the queue, de-queue items and add them to this indexq.

        The returned JoinableQueue object can be safely passed to multiple worker processes to populate it with data.

        To indicate that you are done writing the data to the queue, pass in the sentinel value ('STOP' by default).

        Make sure you call join_indexer() after you are done to close out the queue and join the worker.
        '''
        self.in_q = JoinableQueue()
        self.indexer_process = Process(target=self._indexer_process, args=(self.in_q, sentinel))
        self.indexer_process.daemon = False
        self.indexer_process.start()
        return self.in_q