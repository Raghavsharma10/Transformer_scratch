def register_file_monitor(filename, target=None):
    """Maps a specific file/directory modification event to a signal.

    :param str|unicode filename: File or a directory to watch for its modification.

    :param int|Signal|str|unicode target: Existing signal to raise
        or Signal Target to register signal implicitly.

        Available targets:

            * ``workers``  - run the signal handler on all the workers
            * ``workerN`` - run the signal handler only on worker N
            * ``worker``/``worker0`` - run the signal handler on the first available worker
            * ``active-workers`` - run the signal handlers on all the active [non-cheaped] workers

            * ``mules`` - run the signal handler on all of the mules
            * ``muleN`` - run the signal handler on mule N
            * ``mule``/``mule0`` - run the signal handler on the first available mule

            * ``spooler`` - run the signal on the first available spooler
            * ``farmN/farm_XXX``  - run the signal handler in the mule farm N or named XXX

    :raises ValueError: If unable to register monitor.
    """
    return _automate_signal(target, func=lambda sig: uwsgi.add_file_monitor(int(sig), filename))