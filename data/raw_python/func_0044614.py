def execute(self, operation, parameters=()):
        """
        Wraps execute method to record the query, execution duration and
        stackframe.
        """
        __traceback_hide__ = True  # NOQ

        # Time the exection of the query

        start = time.time()
        try:
            return self.cursor.execute(operation, parameters)
        finally:
            end = time.time()

            # Save the data
            data = {
                'name': operation,
                'args': parameters,
                'start': start,
                'end': end,
            }
            self._record(data)