def set_memory_params(self, ksm_interval=None, no_swap=None):
        """Set memory related parameters.

        :param int ksm_interval: Kernel Samepage Merging frequency option, that can reduce memory usage.
            Accepts a number of requests (or master process cycles) to run page scanner after.

            .. note:: Linux only.

            * http://uwsgi.readthedocs.io/en/latest/KSM.html

        :param bool no_swap: Lock all memory pages avoiding swapping.

        """
        self._set('ksm', ksm_interval)
        self._set('never_swap', no_swap, cast=bool)

        return self._section