def format(self, subtitles):
        """Turn a string containing the subs xml document into the formatted
        subtitle string

        @param str|crunchyroll.models.StyledSubtitle sub_xml_text
        @return str
        """
        logger.debug('Formatting subtitles (id=%s) with %s',
            subtitles.id, self.__class__.__name__)
        return self._format(subtitles).encode('utf-8')