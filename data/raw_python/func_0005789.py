def set_emergency_params(
            self, workers_step=None, idle_cycles_max=None, queue_size=None, queue_nonzero_delay=None):
        """Sets busyness algorithm emergency workers related params.

        Emergency workers could be spawned depending upon uWSGI backlog state.

        .. note:: These options are Linux only.

        :param int workers_step: Number of emergency workers to spawn. Default: 1.

        :param int idle_cycles_max: Idle cycles to reach before stopping an emergency worker. Default: 3.

        :param int queue_size: Listen queue (backlog) max size to spawn an emergency worker. Default: 33.

        :param int queue_nonzero_delay: If the request listen queue is > 0 for more than given amount of seconds
            new emergency workers will be spawned. Default: 60.

        """
        self._set('cheaper-busyness-backlog-step', workers_step)
        self._set('cheaper-busyness-backlog-multiplier', idle_cycles_max)
        self._set('cheaper-busyness-backlog-alert', queue_size)
        self._set('cheaper-busyness-backlog-nonzero', queue_nonzero_delay)

        return self