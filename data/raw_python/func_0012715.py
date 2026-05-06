def analyze(self, text):
        """
        Runs a line of text through MeCab, and returns the results as a
        list of lists ("records") that contain the MeCab analysis of each
        word.
        """
        try:
            self.process  # make sure things are loaded
            text = render_safe(text).replace('\n', ' ').lower()
            results = []
            for chunk in string_pieces(text):
                self.send_input((chunk + '\n').encode('utf-8'))
                while True:
                    out_line = self.receive_output_line().decode('utf-8')
                    if out_line == 'EOS\n':
                        break

                    word, info = out_line.strip('\n').split('\t')
                    record_parts = [word] + info.split(',')

                    # Pad the record out to have 10 parts if it doesn't
                    record_parts += [None] * (10 - len(record_parts))
                    record = MeCabRecord(*record_parts)

                    # special case for detecting nai -> n
                    if (record.surface == 'ん' and
                        record.conjugation == '不変化型'):
                        # rebuild the record so that record.root is 'nai'
                        record_parts[MeCabRecord._fields.index('root')] = 'ない'
                        record = MeCabRecord(*record_parts)

                    results.append(record)
            return results
        except ProcessError:
            self.restart_process()
            return self.analyze(text)