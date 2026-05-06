def switch_into_lazy_mode(self, affect_master=None):
        """Load apps in workers instead of master.

        This option may have memory usage implications
        as Copy-on-Write semantics can not be used.

        .. note:: Consider using ``touch_chain_reload`` option in ``workers`` basic params
            for lazy apps reloading.

        :param bool affect_master: If **True** only workers will be
          reloaded by uWSGI's reload signals; the master will remain alive.

          .. warning:: uWSGI configuration changes are not picked up on reload by the master.


        """
        self._set('lazy' if affect_master else 'lazy-apps', True, cast=bool)

        return self._section