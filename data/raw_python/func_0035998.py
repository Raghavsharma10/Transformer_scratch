def status(self):
        """Get the status of the daemon."""
        if self.pidfile is None:
            raise DaemonError('Cannot get status of daemon without PID file')

        pid = self._read_pidfile()
        if pid is None:
            self._emit_message(
                '{prog} -- not running\n'.format(prog=self.prog))
            sys.exit(1)

        proc = psutil.Process(pid)
        # Default data
        data = {
            'prog': self.prog,
            'pid': pid,
            'status': proc.status(),
            'uptime': '0m',
            'cpu': 0.0,
            'memory': 0.0,
        }

        # Add up all the CPU and memory usage of all the
        # processes in the process group
        pgid = os.getpgid(pid)
        for gproc in psutil.process_iter():
            try:
                if os.getpgid(gproc.pid) == pgid and gproc.pid != 0:
                    data['cpu'] += gproc.cpu_percent(interval=0.1)
                    data['memory'] += gproc.memory_percent()
            except (psutil.Error, OSError):
                continue

        # Calculate the uptime and format it in a human-readable but
        # also machine-parsable format
        try:
            uptime_mins = int(round((time.time() - proc.create_time()) / 60))
            uptime_hours, uptime_mins = divmod(uptime_mins, 60)
            data['uptime'] = str(uptime_mins) + 'm'
            if uptime_hours:
                uptime_days, uptime_hours = divmod(uptime_hours, 24)
                data['uptime'] = str(uptime_hours) + 'h ' + data['uptime']
                if uptime_days:
                    data['uptime'] = str(uptime_days) + 'd ' + data['uptime']
        except psutil.Error:
            pass

        template = ('{prog} -- pid: {pid}, status: {status}, '
                    'uptime: {uptime}, %cpu: {cpu:.1f}, %mem: {memory:.1f}\n')
        self._emit_message(template.format(**data))