def init(self):
        """Deferred initialization b/c the object con be constructed without a valid source"""
        from itertools import chain

        def maybe_int(v):
            try:
                return int(v)
            except ValueError:
                return None

        if not self.initialized:

            self.data_start_line = 1
            self.data_end_line = None
            self.header_lines = [0]

            if self.source.start_line:
                self.data_start_line = self.source.start_line
            if self.source.end_line:
                self.data_end_line = self.source.end_line
            if self.source.header_lines:
                self.header_lines = list(map(maybe_int, self.source.header_lines))
            if self.source.comment_lines:
                self.header_comment_lines = list(map(maybe_int, self.source.comment_lines))

            max_header_line = max(chain(self.header_comment_lines, self.header_lines))

            if self.data_start_line <= max_header_line:
                self.data_start_line = max_header_line + 1

            if not self.header_comment_lines:
                min_header_line = min(chain(self.header_lines))
                if min_header_line:
                    self.header_comment_lines = list(range(0, min_header_line))

            self.headers = []
            self.header_comments = []
            self.footers = []

            self.initialized = True
            self.i = 0