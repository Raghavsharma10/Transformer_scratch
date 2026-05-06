def decide(self):
        """ Choose the next command to execute, and its parameters, based on the current
            state.
        """

        next_command_name = random.choice(self.COMMAND_MAP[self.state['last_command']])
        param = ''
        if next_command_name == 'cd':
            try:
                param = random.choice(self.state['dir_list'])
            except IndexError:
                next_command_name = 'ls'

        elif next_command_name == 'uname':
            opts = 'asnrvmpio'
            start = random.randint(0, len(opts) - 2)
            end = random.randint(start + 1, len(opts) - 1)
            param = '-{}'.format(opts[start:end])
        elif next_command_name == 'ls':
            if random.randint(0, 1):
                param = '-l'
        elif next_command_name == 'cat':
            try:
                param = random.choice(self.state['file_list'])
            except IndexError:
                param = ''.join(random.choice(string.lowercase) for x in range(3))
        elif next_command_name == 'echo':
            param = random.choice([
                '$http_proxy',
                '$https_proxy',
                '$ftp_proxy',
                '$BROWSER',
                '$EDITOR',
                '$SHELL',
                '$PAGER'
            ])
        elif next_command_name == 'sudo':
            param = random.choice([
                'pm-hibernate',
                'shutdown -h',
                'vim /etc/httpd.conf',
                'vim /etc/resolve.conf',
                'service network restart',
                '/etc/init.d/network-manager restart',
            ])
        return next_command_name, param