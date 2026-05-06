def stack_call(self, *args):
        """Stacks a redis command inside the object.

        The syntax is the same than the call() method a Client class.

        Args:
            *args: full redis command as variable length argument list.

        Examples:
            >>> pipeline = Pipeline()
            >>> pipeline.stack_call("HSET", "key", "field", "value")
            >>> pipeline.stack_call("PING")
            >>> pipeline.stack_call("INCR", "key2")
        """
        self.pipelined_args.append(args)
        self.number_of_stacked_calls = self.number_of_stacked_calls + 1