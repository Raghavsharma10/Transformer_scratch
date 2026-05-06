def create_session():
        """
        shell = CommandShell('http://192.168.145.132:5985/wsman', 'Administrator', 'Pa55w0rd')
        """
        shell = CommandShell('http://192.168.137.238:5985/wsman', 'Administrator', 'Pa55w0rd')
        shell.open()
        command_id = shell.run('ipconfig', ['/all'])
        (stdout, stderr, exit_code) = shell.receive(command_id)
        sys.stdout.write(stdout.strip() + '\r\n')
        shell.close()

        return None