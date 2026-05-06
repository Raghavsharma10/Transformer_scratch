def _open(self):
        """
        Opens the next current file with proper settings for encoding and delimiter.
        """
        self._sample = None

        formatting_parameters0 = {'encoding':        'auto',
                                  'delimiter':       'auto',
                                  'line_terminator': 'auto',
                                  'escape_char':     '\\',
                                  'quote_char':      '"'}
        formatting_parameters1 = self._helper.pass1(self._filename, formatting_parameters0)
        self._formatting_parameters = formatting_parameters1

        # Detect encoding.
        if formatting_parameters1['encoding'] == 'auto':
            self._get_sample('rb', None)
            self._detect_encoding()

        # Detect delimiter.
        if formatting_parameters1['delimiter'] == 'auto':
            self._get_sample('rt', formatting_parameters1['encoding'])
            self._detect_delimiter()

        # Detect line terminators.
        if formatting_parameters1['line_terminator'] == 'auto':
            if not self._sample:
                self._get_sample('rt', formatting_parameters1['encoding'])
            self._detect_line_ending()

        self._formatting_parameters = self._helper.pass2(self._filename,
                                                         self._formatting_parameters,
                                                         formatting_parameters1)

        self._open_file('rt', formatting_parameters1['encoding'])
        self._csv_reader = csv.reader(self._file,
                                      delimiter=self._formatting_parameters['delimiter'],
                                      escapechar=self._formatting_parameters['escape_char'],
                                      lineterminator=self._formatting_parameters['line_terminator'],
                                      quotechar=self._formatting_parameters['quote_char'])  # Ignored

        self._sample = None