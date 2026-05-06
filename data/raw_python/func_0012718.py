def analyze(self, text):
        """
        Run text through the external process, and get a list of lists
        ("records") that contain the analysis of each word.
        """
        try:
            text = render_safe(text).strip()
            if not text:
                return []
            chunks = text.split('\n')
            results = []
            for chunk_text in chunks:
                if chunk_text.strip():
                    textbytes = (chunk_text + '\n').encode('utf-8')
                    self.send_input(textbytes)
                    out_line = ''
                    while True:
                        out_line = self.receive_output_line()
                        out_line = out_line.decode('utf-8')

                        if out_line == '\n':
                            break

                        record = out_line.strip('\n').split(' ')
                        results.append(record)
            return results
        except ProcessError:
            self.restart_process()
            return self.analyze(text)