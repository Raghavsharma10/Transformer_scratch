def update_note(self, note_id, revision, content):
        ''' Updates the note with the given ID to have the given content '''
        return notes_endpoint.update_note(self, note_id, revision, content)