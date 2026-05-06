def execute_command(self, *args, **options):
        """Execute a command and return a parsed response"""
        pool = self.connection_pool
        command_name = args[0]
        for i in _xrange(self.execution_attempts):
            connection = pool.get_connection(command_name, **options)
            try:
                connection.send_command(*args)
                res = self.parse_response(connection, command_name, **options)
                pool.release(connection)
                return res
            except ConnectionError:
                pool.purge(connection)
                if i >= self.execution_attempts - 1:
                    raise