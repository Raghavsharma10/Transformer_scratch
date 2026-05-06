def log(self, facility, level, text, pid=False):
        """Send the message text to all registered hosts.

        The facility and level will be used to create the packet's PRI
        part. The HEADER will be automatically determined from the
        current time and hostname. The MSG will be set from the
        running program's name and the text parameter.

        This is the simplest way to use reSyslog.Syslog, creating log
        messages containing the current time, hostname, program name,
        etc. This is how you do it::

            logger = syslog.Syslog()
            logger.add_host("localhost")
            logger.log(Facility.USER, Level.INFO, "Hello World")

        If pid is True the process ID will be prepended to the text
        parameter, enclosed in square brackets and followed by a
        colon.

        """
        pri = PRI(facility, level)
        header = HEADER()
        if pid:
            msg = MSG(content=text, pid=os.getpid())
        else:
            msg = MSG(content=text)
        packet = Packet(pri, header, msg)
        self._send_packet_to_hosts(packet)