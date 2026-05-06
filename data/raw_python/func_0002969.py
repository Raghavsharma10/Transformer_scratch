def make_back_notes(self, body):
        """
        The notes element in PLoS articles can be employed for posting notices
        of corrections or adjustments in proof. The <notes> element has a very
        diverse content model, but PLoS practice appears to be fairly
        consistent: a single <sec> containing a <title> and a <p>
        """
        for notes in self.article.root.xpath('./back/notes'):
            notes_sec = deepcopy(notes.find('sec'))
            notes_sec.tag = 'div'
            notes_sec.attrib['class'] = 'back-notes'
            body.append(notes_sec)