def query(self, command):
        """
        WILL BLOCK CALLING THREAD UNTIL THE command IS COMPLETED
        :param command: COMMAND FOR SQLITE
        :return: list OF RESULTS
        """
        if self.closed:
            Log.error("database is closed")

        signal = _allocate_lock()
        signal.acquire()
        result = Data()
        trace = extract_stack(1) if self.get_trace else None

        if self.get_trace:
            current_thread = Thread.current()
            with self.locker:
                for t in self.available_transactions:
                    if t.thread is current_thread:
                        Log.error(DOUBLE_TRANSACTION_ERROR)

        self.queue.add(CommandItem(command, result, signal, trace, None))
        signal.acquire()

        if result.exception:
            Log.error("Problem with Sqlite call", cause=result.exception)
        return result