def type_id(self):
        """
        Shortcut to retrieving the ContentType id of the model.
        """
        try:
            return ContentType.objects.get_for_model(self.model, for_concrete_model=False).id
        except DatabaseError as e:
            raise DatabaseError("Unable to fetch ContentType object, is a plugin being registered before the initial syncdb? (original error: {0})".format(str(e)))