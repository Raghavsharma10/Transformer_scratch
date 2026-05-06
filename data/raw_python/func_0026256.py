def add_contact(self):
        """ Create a contact with using the email on the list. """
        self.api.lists.addcontact(
            contact=self.cleaned_data['email'], id=self.list_id, method='POST')