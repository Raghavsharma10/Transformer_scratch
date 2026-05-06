def read_time(self, content):
        """ Core function used to generate the read_time for content.

        Parameters:
            :param content: Instance of pelican.content.Content

        Returns:
            None
        """
        if get_class_name(content) in self.content_type_supported:
            # Exit if readtime is already set
            if hasattr(content, 'readtime'):
                return None

            default_lang_conf = self.lang_settings['default']
            lang_conf = self.lang_settings.get(content.lang, default_lang_conf)
            avg_reading_wpm = lang_conf['wpm']

            num_words = len(content._content.split())
            # Floor division so we don't have to convert float -> int
            minutes = num_words // avg_reading_wpm
            # Get seconds to read, then subtract our minutes as seconds from
            # the time to get remainder seconds
            seconds = int((num_words / avg_reading_wpm * 60) - (minutes * 60))

            minutes_str = self.pluralize(
                minutes,
                lang_conf['min_singular'],
                lang_conf['min_plural']
            )

            seconds_str = self.pluralize(
                seconds,
                lang_conf['sec_singular'],
                lang_conf['sec_plural']
            )

            content.readtime = minutes
            content.readtime_string = minutes_str
            content.readtime_with_seconds = (minutes, seconds,)
            content.readtime_string_with_seconds = "{}, {}".format(
                minutes_str, seconds_str)