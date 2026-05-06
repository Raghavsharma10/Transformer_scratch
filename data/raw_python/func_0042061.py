def _parse_current_member(self, previous_rank, values):
        """
        Parses the column texts of a member row into a member dictionary.

        Parameters
        ----------
        previous_rank: :class:`dict`[int, str]
            The last rank present in the rows.
        values: tuple[:class:`str`]
            A list of row contents.
        """
        rank, name, vocation, level, joined, status = values
        rank = previous_rank[1] if rank == " " else rank
        title = None
        previous_rank[1] = rank
        m = title_regex.match(name)
        if m:
            name = m.group(1)
            title = m.group(2)
        self.members.append(GuildMember(name, rank, title, int(level), vocation, joined=joined,
                                        online=status == "online"))