def referenceframe(self, event):
        """Handles navigational reference frame updates.
        These are necessary to assign geo coordinates to alerts and other
        misc things.

        :param event with incoming referenceframe message
        """

        self.log("Got a reference frame update! ", event, lvl=events)

        self.reference_frame = event.data