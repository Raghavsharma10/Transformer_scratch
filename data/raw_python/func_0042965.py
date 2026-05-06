def wait_pid(self):
        "Wait for the daemon, get its PID and a control-socket connection."
        while True:
            try:
                fp = open(self.pidfile)
            except IOError:
                time.sleep(0.1)
                continue
            try:
                fp.seek(0)
                pidstr = fp.read()
                self.pid = int(pidstr)
            except ValueError:
                time.sleep(0.5)
                continue	# Avoid race condition -- PID not yet written
            fp.close()
            break