def set_mules_params(
            self, mules=None, touch_reload=None, harakiri_timeout=None, farms=None, reload_mercy=None,
            msg_buffer=None, msg_buffer_recv=None):
        """Sets mules related params.

        http://uwsgi.readthedocs.io/en/latest/Mules.html

        Mules are worker processes living in the uWSGI stack but not reachable via socket connections,
        that can be used as a generic subsystem to offload tasks.

        :param int|list mules: Add the specified mules or number of mules.

        :param str|list touch_reload: Reload mules if the specified file is modified/touched.

        :param int harakiri_timeout: Set harakiri timeout for mule tasks.

        :param list[MuleFarm] farms: Mule farms list.

            Examples:
                * cls_mule_farm('first', 2)
                * cls_mule_farm('first', [4, 5])

        :param int reload_mercy: Set the maximum time (in seconds) a mule can take
            to reload/shutdown. Default: 60.

        :param int msg_buffer: Set mule message buffer size (bytes) given
            for mule message queue.

        :param int msg_buffer: Set mule message recv buffer size (bytes).

        """
        farms = farms or []

        next_mule_number = 1
        farm_mules_count = 0

        for farm in farms:

            if isinstance(farm.mule_numbers, int):
                farm.mule_numbers = list(range(next_mule_number, next_mule_number + farm.mule_numbers))
                next_mule_number = farm.mule_numbers[-1] + 1

            farm_mules_count += len(farm.mule_numbers)

            self._set('farm', farm, multi=True)

        if mules is None and farm_mules_count:
            mules = farm_mules_count

        if isinstance(mules, int):
            self._set('mules', mules)

        elif isinstance(mules, list):

            for mule in mules:
                self._set('mule', mule, multi=True)

        self._set('touch-mules-reload', touch_reload, multi=True)

        self._set('mule-harakiri', harakiri_timeout)
        self._set('mule-reload-mercy', reload_mercy)

        self._set('mule-msg-size', msg_buffer)
        self._set('mule-msg-recv-size', msg_buffer_recv)

        return self._section