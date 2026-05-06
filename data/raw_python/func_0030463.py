def requestAvatar(self, avatarId, mind, *interfaces):
        """
        Create Adder avatars for any IBoxReceiver request.
        """
        if IBoxReceiver in interfaces:
            return (IBoxReceiver, Adder(avatarId), lambda: None)
        raise NotImplementedError()