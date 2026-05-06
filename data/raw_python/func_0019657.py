def open(self, host=None, port=0, socket_file=None, 
             timeout=socket.getdefaulttimeout()):
        """Connect to a host.

        With a host argument, it connects the instance using TCP; port number 
        and timeout are optional, socket_file must be None. The port number
        defaults to the standard telnet port (23).
        
        With a socket_file argument, it connects the instance using
        named socket; timeout is optional and host must be None.

        Don't try to reopen an already connected instance.
        
        """
        self.socket_file = socket_file
        if host is not None:
            if sys.version_info[:2] >= (2,6):
                telnetlib.Telnet.open(self, host, port, timeout)
            else:
                telnetlib.Telnet.open(self, host, port)
        elif socket_file is not None:
            self.eof = 0
            self.host = host
            self.port = port
            self.timeout = timeout
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(timeout)
            self.sock.connect(socket_file)
        else:
            raise TypeError("Either host or socket_file argument is required.")