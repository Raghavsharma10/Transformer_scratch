def register_timer_rb(period, repeat=None, target=None):
    """Add a red-black timer (based on black-red tree).

        .. code-block:: python

            @register_timer_rb(3)
            def repeat():
                do()

    :param int period: The interval (seconds) at which the signal is raised.

    :param int repeat: How many times to repeat. Default: None - infinitely.

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

    :rtype: bool|callable

    :raises ValueError: If unable to add timer.
    """
    return _automate_signal(target, func=lambda sig: uwsgi.add_rb_timer(int(sig), period, repeat or 0))