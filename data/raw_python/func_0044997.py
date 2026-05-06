def run(path, code, params=None, ignore=None, select=None, **meta):
        """Pylint code checking.

        :return list: List of errors.
        """
        logger.debug('Start pylint')

        clear_cache = params.pop('clear_cache', False)
        if clear_cache:
            MANAGER.astroid_cache.clear()

        class Reporter(BaseReporter):

            def __init__(self):
                self.errors = []
                super(Reporter, self).__init__()

            def _display(self, layout):
                pass

            def handle_message(self, msg):
                self.errors.append(dict(
                    lnum=msg.line,
                    col=msg.column,
                    text="%s %s" % (msg.msg_id, msg.msg),
                    type=msg.msg_id[0]
                ))

        params = _Params(ignore=ignore, select=select, params=params)
        logger.debug(params)

        reporter = Reporter()

        try:
            Run([path] + params.to_attrs(), reporter=reporter, do_exit=False)
        except TypeError:
            # support pylint<2.0
            # see https://github.com/PyCQA/pylint/commit/4210ef9b8c5d9e7b33ff0542683f18b8031193fa
            import pylint
            if pylint.__version__.split('.')[0] != '1':
                raise
            Run([path] + params.to_attrs(), reporter=reporter, exit=False)

        return reporter.errors