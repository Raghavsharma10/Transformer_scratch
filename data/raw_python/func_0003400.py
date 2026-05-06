def main(configpath = None, startup = None, daemon = False, pidfile = None, fork = None):
    """
    The most simple way to start the VLCP framework
    
    :param configpath: path of a configuration file to be loaded
    
    :param startup: startup modules list. If None, `server.startup` in the configuration files
                    is used; if `server.startup` is not configured, any module defined or imported
                    into __main__ is loaded.
    
    :param daemon: if True, use python-daemon to fork and start at background. `python-daemon` must be
                   installed::
                    
                       pip install python-daemon
    
    :param pidfile: if daemon=True, this file is used for the pidfile.
    
    :param fork: use extra fork to start multiple instances
    """
    if configpath is not None:
        manager.loadfrom(configpath)
    if startup is not None:
        manager['server.startup'] = startup
    if not manager.get('server.startup'):
        # No startup modules, try to load from __main__
        startup = []
        import __main__
        for k in dir(__main__):
            m = getattr(__main__, k)
            if isinstance(m, type) and issubclass(m, Module) and m is not Module:
                startup.append('__main__.' + k)
        manager['server.startup'] = startup
    if fork is not None and fork > 1:
        if not hasattr(os, 'fork'):
            raise ValueError('Fork is not supported in this operating system.')
    def start_process():
        s = Server()
        s.serve()
    def main_process():
        if fork is not None and fork > 1:
            import multiprocessing
            from time import sleep
            sub_procs = []
            for i in range(0, fork):
                p = multiprocessing.Process(target = start_process)
                sub_procs.append(p)
            for i in range(0, fork):
                sub_procs[i].start()
            try:
                import signal
                def except_return(sig, frame):
                    raise SystemExit
                signal.signal(signal.SIGTERM, except_return)
                signal.signal(signal.SIGINT, except_return)
                if hasattr(signal, 'SIGHUP'):
                    signal.signal(signal.SIGHUP, except_return)
                while True:
                    sleep(2)
                    for i in range(0, fork):
                        if sub_procs[i].is_alive():
                            break
                    else:
                        break
            finally:
                for i in range(0, fork):
                    if sub_procs[i].is_alive():
                        sub_procs[i].terminate()
                for i in range(0, fork):
                    sub_procs[i].join()
        else:
            start_process()
    if daemon:
        import daemon
        if not pidfile:
            pidfile = manager.get('daemon.pidfile')
        uid = manager.get('daemon.uid')
        gid = manager.get('daemon.gid')
        if gid is None:
            group = manager.get('daemon.group')
            if group is not None:
                import grp
                gid = grp.getgrnam(group)[2]
        if uid is None:
            import pwd
            user = manager.get('daemon.user')
            if user is not None:
                user_pw = pwd.getpwnam(user)
                uid = user_pw.pw_uid
                if gid is None:
                    gid = user_pw.pw_gid
        if uid is not None and gid is None:
            import pwd
            gid = pwd.getpwuid(uid).pw_gid
        if pidfile:
            import fcntl
            class PidLocker(object):
                def __init__(self, path):
                    self.filepath = path
                    self.fd = None
                def __enter__(self):
                    # Create pid file
                    self.fd = os.open(pidfile, os.O_WRONLY | os.O_TRUNC | os.O_CREAT, 0o644)
                    fcntl.lockf(self.fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
                    os.write(self.fd, str(os.getpid()).encode('ascii'))
                    os.fsync(self.fd)
                def __exit__(self, typ, val, tb):
                    if self.fd:
                        try:
                            fcntl.lockf(self.fd, fcntl.LOCK_UN)
                        except Exception:
                            pass
                        os.close(self.fd)
                        self.fd = None
            locker = PidLocker(pidfile)
        else:
            locker = None
        import sys
        # Module loading is related to current path, add it to sys.path
        cwd = os.getcwd()
        if cwd not in sys.path:
            sys.path.append(cwd)
        # Fix path issues on already-loaded modules
        for m in sys.modules.values():
            if getattr(m, '__path__', None):
                m.__path__ = [os.path.abspath(p) for p in m.__path__]
            # __file__ is used for module-relative resource locate
            if getattr(m, '__file__', None):
                m.__file__ = os.path.abspath(m.__file__)
        configs = {'gid':gid,'uid':uid,'pidfile':locker}
        config_filters = ['chroot_directory', 'working_directory', 'umask', 'detach_process',
                          'prevent_core']
        if hasattr(manager, 'daemon'):
            configs.update((k,v) for k,v in manager.daemon.config_value_items() if k in config_filters)
        if not hasattr(os, 'initgroups'):
            configs['initgroups'] = False
        with daemon.DaemonContext(**configs):
            main_process()
    else:
        main_process()