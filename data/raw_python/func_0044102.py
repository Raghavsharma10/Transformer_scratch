def track_heads(self, cmd):
        """Track the repository heads given a CommitCommand.

        :param cmd: the CommitCommand
        :return: the list of parents in terms of commit-ids
        """
        # Get the true set of parents
        if cmd.from_ is not None:
            parents = [cmd.from_]
        else:
            last_id = self.last_ids.get(cmd.ref)
            if last_id is not None:
                parents = [last_id]
            else:
                parents = []
        parents.extend(cmd.merges)

        # Track the heads
        self.track_heads_for_ref(cmd.ref, cmd.id, parents)
        return parents