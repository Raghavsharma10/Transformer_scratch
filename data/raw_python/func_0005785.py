def add_cron_task(
            self, command, weekday=None, month=None, day=None, hour=None, minute=None,
            legion=None, unique=None, harakiri=None):
        """Adds a cron task running the given command on the given schedule.
        http://uwsgi.readthedocs.io/en/latest/Cron.html

        HINTS:
            * Use negative values to say `every`:
                hour=-3  stands for `every 3 hours`

            * Use - (minus) to make interval:
                minute='13-18'  stands for `from minute 13 to 18`

        .. note:: We use cron2 option available since 1.9.11.

        :param str|unicode command: Command to execute on schedule (with or without path).

        :param int|str|unicode weekday: Day of a the week number. Defaults to `each`.
            0 - Sunday  1 - Monday  2 - Tuesday  3 - Wednesday
            4 - Thursday  5 - Friday  6 - Saturday

        :param int|str|unicode month: Month number 1-12. Defaults to `each`.

        :param int|str|unicode day: Day of the month number 1-31. Defaults to `each`.

        :param int|str|unicode hour: Hour 0-23. Defaults to `each`.

        :param int|str|unicode minute: Minute 0-59. Defaults to `each`.

        :param str|unicode legion: Set legion (cluster) name to use this cron command against.
            Such commands are only executed by legion lord node.

        :param bool unique: Marks command as unique. Default to not unique.
            Some commands can take a long time to finish or just hang doing their thing.
            Sometimes this is okay, but there are also cases when running multiple instances
            of the same command can be dangerous.

        :param int harakiri: Enforce a time limit (in seconds) on executed commands.
            If a command is taking longer it will be killed.

        """
        rule = KeyValue(
            locals(),
            keys=['weekday', 'month', 'day', 'hour', 'minute', 'harakiri', 'legion', 'unique'],
            aliases={'weekday': 'week'},
            bool_keys=['unique'],
        )

        self._set('cron2', ('%s %s' % (rule, command)).strip(), multi=True)

        return self._section