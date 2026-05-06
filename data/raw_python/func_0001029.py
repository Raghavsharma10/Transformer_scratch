def get_participants(self, obj):
        """ Allows to define a callback for serializing information about the user. """
        # we set the many to many serialization to False, because we only want it with retrieve requests
        if self.callback is None:
            return [participant.id for participant in obj.participants.all()]
        else:
            # we do not want user information
            return self.callback(obj)