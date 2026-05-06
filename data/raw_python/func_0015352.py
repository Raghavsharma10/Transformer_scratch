def _start_ssh_agent(cls):
        """Starts ssh-agent and returns the environment variables related to it"""
        env = dict()
        stdout = ClHelper.run_command('ssh-agent -s')
        lines = stdout.split('\n')
        for line in lines:
            if not line or line.startswith('echo '):
                continue
            line = line.split(';')[0]
            parts = line.split('=')
            if len(parts) == 2:
                env[parts[0]] = parts[1]
        return env