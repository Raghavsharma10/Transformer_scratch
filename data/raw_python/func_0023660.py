def daemon_factory(path):
    """Create a closure which creates a running daemon.

    We need to create a closure that contains the correct path the daemon should
    be started with. This is needed as the `Daemonize` library
    requires a callable function for daemonization and doesn't accept any arguments.
    This function cleans up sockets and output files in case we encounter any exceptions.
    """
    def start_daemon():
        root_dir = path
        config_dir = os.path.join(root_dir, '.config/pueue')
        try:
            daemon = Daemon(root_dir=root_dir)
            daemon.main()
        except KeyboardInterrupt:
            print('Keyboard interrupt. Shutting down')
            daemon.stop_daemon()
        except Exception:
            try:
                daemon.stop_daemon()
            except Exception:
                pass
            cleanup(config_dir)
            raise
    return start_daemon