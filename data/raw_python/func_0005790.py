def set_basic_params(
            self, spawn_on_request=None,
            cheaper_algo=None, workers_min=None, workers_startup=None, workers_step=None):
        """
        :param bool spawn_on_request: Spawn workers only after the first request.

        :param Algo cheaper_algo: The algorithm object to be used used for adaptive process spawning.
            Default: ``spare``. See ``.algorithms``.

        :param int workers_min: Minimal workers count. Enables cheaper mode (adaptive process spawning).

            .. note:: Must be lower than max workers count.

        :param int workers_startup: The number of workers to be started when starting the application.
            After the app is started the algorithm can stop or start workers if needed.

        :param int workers_step: Number of additional processes to spawn at a time if they are needed,

        """
        self._set('cheap', spawn_on_request, cast=bool)

        if cheaper_algo:
            self._set('cheaper-algo', cheaper_algo.name)

            if cheaper_algo.plugin:
                self._section.set_plugins_params(plugins=cheaper_algo.plugin)

            cheaper_algo._contribute_to_opts(self)

        self._set('cheaper', workers_min)
        self._set('cheaper-initial', workers_startup)
        self._set('cheaper-step', workers_step)

        return self._section