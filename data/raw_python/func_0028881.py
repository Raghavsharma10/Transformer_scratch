def _is_already_running(self):
        """Check to see if the process is running, first looking for a pidfile,
        then shelling out in either case, removing a pidfile if it exists but
        the process is not running.

        """
        # Look for the pidfile, if exists determine if the process is alive
        pidfile = self._get_pidfile_path()
        if os.path.exists(pidfile):
            pid = open(pidfile).read().strip()
            try:
                os.kill(int(pid), 0)
                sys.stderr.write('Process already running as pid # %s\n' % pid)
                return True
            except OSError as error:
                LOGGER.debug('Found pidfile, no process # %s', error)
                os.unlink(pidfile)

        # Check the os for a process that is not this one that looks the same
        pattern = ' '.join(sys.argv)
        pattern = '[%s]%s' % (pattern[0], pattern[1:])
        try:
            output = subprocess.check_output('ps a | grep "%s"' % pattern,
                                             shell=True)
        except AttributeError:
            # Python 2.6
            stdin, stdout, stderr = os.popen3('ps a | grep "%s"' % pattern)
            output = stdout.read()
        except subprocess.CalledProcessError:
            return False
        pids = [int(pid) for pid in (re.findall(r'^([0-9]+)\s',
                                                output.decode('latin-1')))]
        if os.getpid() in pids:
            pids.remove(os.getpid())
        if not pids:
            return False
        if len(pids) == 1:
            pids = pids[0]
        sys.stderr.write('Process already running as pid # %s\n' % pids)
        return True