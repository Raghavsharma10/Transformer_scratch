def _parse_invited_member(self, values):
        """
        Parses the column texts of an invited row into a invited dictionary.

        Parameters
        ----------
        values: tuple[:class:`str`]
            A list of row contents.
        """
        name, date = values
        if date != "Invitation Date":
            self.invites.append(GuildInvite(name, date))