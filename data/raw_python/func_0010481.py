def from_file(cls, filepath, chunk_size=None):
        """
        Try do determine the language of a text file.
        :param filepath: string file path
        :param chunk_size: amount of bytes of file to read to determine language
        :return: Language instance if detection succeeded, otherwise return UnknownLanguage
        """
        log.debug('Language.from_file: "{}", chunk={} ...'.format(filepath, chunk_size))
        with filepath.open('rb') as f:
            data = f.read(-1 if chunk_size is None else chunk_size)
        data_ascii = asciify(data)
        lang_xx = langdetect_detect(data_ascii)
        lang = cls.from_xx(lang_xx)
        log.debug('... result language={}'.format(lang))
        return lang