def add_scheduled_job(self, text_source, cron_args, text_format, title, author, summary=None,
                          synthesizer='watson', synth_args=None, sentence_break='. '):
        """
        Add and start a new scheduled job to dynamically generate podcasts.

        Note: scheduling will end when the process ends. This works best when run
        inside an existing application.

        :param text_source:
            A function that generates podcast text. Examples: a function that
            opens a file with today's date as a filename or a function that
            requests a specific url and extracts the main text.
            Also see :meth:`Episode`.
        :param cron_args:
            A dictionary of cron parameters. Keys can be: 'year', 'month',
            'day', 'week', 'day_of_week', 'hour', 'minute' and 'second'. Keys
            that are not specified will be parsed as 'any'/'*'.
        :param text_format:
            See :meth:`Episode`.
        :param title:
            See :meth:`Episode`. Since titles need to be unique, a
            timestamp will be appended to the title for each episode.
        :param author:
            See :meth:`Episode`.
        :param summary:
            See :meth:`Episode`.
        :param publish_date:
            See :meth:`Episode`.
        :param synthesizer:
            See :meth:`typecaster.utils.text_to_speech`.
        :param synth_args:
            See :meth:`typecaster.utils.text_to_speech`.
        :param sentence_break:
            See :meth:`typecaster.utils.text_to_speech`.
        """
        if not callable(text_source):
            raise TypeError('Argument "text" must be a function')

        def add_episode():
            episode_text = text_source()
            episode_title = title + '_' + datetime.utcnow().strftime('%Y%m%d%H%M%S')

            self.add_episode(episode_text, text_format, episode_title, author, summary, datetime.utcnow(), synthesizer, synth_args, sentence_break)

        self.scheduled_jobs[title] = self._scheduler.add_job(add_episode, 'cron', id=title, **cron_args)

        if not self._scheduler.running:
            self._scheduler.start()