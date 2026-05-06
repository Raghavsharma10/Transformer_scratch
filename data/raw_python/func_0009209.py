def get_note(self, note_id):
        """
        Get a loan note that you've invested in by ID

        Parameters
        ----------
        note_id : int
            The note ID

        Returns
        -------
        dict
            A dictionary representing the matching note or False

        Examples
        --------
            >>> from lendingclub import LendingClub
            >>> lc = LendingClub(email='test@test.com', password='secret123')
            >>> lc.authenticate()
            True
            >>> notes = lc.my_notes()                  # Get the first 100 loan notes
            >>> len(notes['loans'])
            100
            >>> notes['total']                          # See the total number of loan notes you have
            630
            >>> notes = lc.my_notes(start_index=100)   # Get the next 100 loan notes
            >>> len(notes['loans'])
            100
            >>> notes = lc.my_notes(get_all=True)       # Get all notes in one request (may be slow)
            >>> len(notes['loans'])
            630
        """

        index = 0
        while True:
            notes = self.my_notes(start_index=index, sort_by='noteId')

            if notes['result'] != 'success':
                break

            # If the first note has a higher ID, we've passed it
            if notes['loans'][0]['noteId'] > note_id:
                break

            # If the last note has a higher ID, it could be in this record set
            if notes['loans'][-1]['noteId'] >= note_id:
                for note in notes['loans']:
                    if note['noteId'] == note_id:
                        return note

            index += 100

        return False