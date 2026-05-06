def set_mode_broodlord_params(
            self, zerg_count=None,
            vassal_overload_sos_interval=None, vassal_queue_items_sos=None):
        """This mode is a way for a vassal to ask for reinforcements to the Emperor.

        Reinforcements are new vassals spawned on demand generally bound on the same socket.

        .. warning:: If you are looking for a way to dynamically adapt the number
            of workers of an instance, check the Cheaper subsystem - adaptive process spawning mode.

            *Broodlord mode is for spawning totally new instances.*

        :param int zerg_count: Maximum number of zergs to spawn.

        :param int vassal_overload_sos_interval: Ask emperor for reinforcement when overloaded.
            Accepts the number of seconds to wait between asking for a new reinforcements.

        :param int vassal_queue_items_sos: Ask emperor for sos if listen queue (backlog) has more
            items than the value specified

        """
        self._set('emperor-broodlord', zerg_count)
        self._set('vassal-sos', vassal_overload_sos_interval)
        self._set('vassal-sos-backlog', vassal_queue_items_sos)

        return self._section