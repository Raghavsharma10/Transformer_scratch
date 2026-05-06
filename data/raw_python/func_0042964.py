def spawn(self, options, port, background=False, prefix=""):
        "Spawn a daemon instance."
        self.spawncmd = None

	# Look for gpsd in GPSD_HOME env variable
        if os.environ.get('GPSD_HOME'):
            for path in os.environ['GPSD_HOME'].split(':'):
                _spawncmd = "%s/gpsd" % path
                if os.path.isfile(_spawncmd) and os.access(_spawncmd, os.X_OK):
                    self.spawncmd = _spawncmd
                    break

	# if we could not find it yet try PATH env variable for it
        if not self.spawncmd:
            if not '/usr/sbin' in os.environ['PATH']:
                os.environ['PATH']=os.environ['PATH'] + ":/usr/sbin"
            for path in os.environ['PATH'].split(':'):
                _spawncmd = "%s/gpsd" % path
                if os.path.isfile(_spawncmd) and os.access(_spawncmd, os.X_OK):
                    self.spawncmd = _spawncmd
                    break

        if not self.spawncmd:
            raise DaemonError("Cannot execute gpsd: executable not found. Set GPSD_HOME env variable")
        # The -b option to suppress hanging on probe returns is needed to cope
        # with OpenBSD (and possibly other non-Linux systems) that don't support
        # anything we can use to implement the FakeGPS.read() method
        self.spawncmd += " -b -N -S %s -F %s -P %s %s" % (port, self.control_socket, self.pidfile, options)
        if prefix:
            self.spawncmd = prefix + " " + self.spawncmd.strip()
        if background:
            self.spawncmd += " &"
        status = os.system(self.spawncmd)
        if os.WIFSIGNALED(status) or os.WEXITSTATUS(status):
            raise DaemonError("daemon exited with status %d" % status)