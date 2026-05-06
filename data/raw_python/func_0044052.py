def commit_handler(self, cmd):
        """Process a CommitCommand."""
        # These pass through if they meet the filtering conditions
        interesting_filecmds = self._filter_filecommands(cmd.iter_files)
        if interesting_filecmds or not self.squash_empty_commits:
            # If all we have is a single deleteall, skip this commit
            if len(interesting_filecmds) == 1 and isinstance(
                interesting_filecmds[0], commands.FileDeleteAllCommand):
                pass
            else:
                # Remember just the interesting file commands
                self.keep = True
                cmd.file_iter = iter(interesting_filecmds)

                # Record the referenced blobs
                for fc in interesting_filecmds:
                    if isinstance(fc, commands.FileModifyCommand):
                        if (fc.dataref is not None and
                            not stat.S_ISDIR(fc.mode)):
                            self.referenced_blobs.append(fc.dataref)

                # Update from and merges to refer to commits in the output
                cmd.from_ = self._find_interesting_from(cmd.from_)
                cmd.merges = self._find_interesting_merges(cmd.merges)
        else:
            self.squashed_commits.add(cmd.id)

        # Keep track of the parents
        if cmd.from_ and cmd.merges:
            parents = [cmd.from_] + cmd.merges
        elif cmd.from_:
            parents = [cmd.from_]
        else:
            parents = None
        if cmd.mark is not None:
            self.parents[b':' + cmd.mark] = parents