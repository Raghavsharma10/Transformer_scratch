def _feed(self, cube, data_sources, data_sinks, global_iter_args):
        """ Feed stub """
        try:
            self._feed_impl(cube, data_sources, data_sinks, global_iter_args)
        except Exception as e:
            montblanc.log.exception("Feed Exception")
            raise