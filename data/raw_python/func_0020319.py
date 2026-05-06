def execute_command(self, cmnd, *args, **options):
        "Execute a command and return a parsed response"
        args, options = self.preprocess_command(cmnd, *args, **options)
        return self.client.execute_command(cmnd, *args, **options)