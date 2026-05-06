def save(self, notes=None):
        '''Save all changes back to Redmine with optional notes.'''
        # Capture the notes if given
        if notes:
            self._changes['notes'] = notes

        # Call the base-class save function
        super(Issue, self).save()