def add_chapter(self, title):
        '''
        Adds a new chapter to the report.

        :param str title: Title of the chapter.
        '''
        chap_id = 'chap%s' % self.chap_counter
        self.chap_counter += 1
        self.sidebar += '<a href="#%s" class="list-group-item">%s</a>\n' % (
            chap_id, title)
        self.body += '<h1 id="%s">%s</h1>\n' % (chap_id, title)