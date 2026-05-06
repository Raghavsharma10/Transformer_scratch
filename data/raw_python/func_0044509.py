def commit_handler(self, cmd):
        """Process a CommitCommand."""
        self.cmd_counts[cmd.name] += 1
        self.committers.add(cmd.committer)
        if cmd.author is not None:
            self.separate_authors_found = True
        for fc in cmd.iter_files():
            self.file_cmd_counts[fc.name] += 1
            if isinstance(fc, commands.FileModifyCommand):
                if fc.mode & 0o111:
                    self.executables_found = True
                if stat.S_ISLNK(fc.mode):
                    self.symlinks_found = True
                if fc.dataref is not None:
                    if fc.dataref[0] == ':':
                        self._track_blob(fc.dataref)
                    else:
                        self.sha_blob_references = True
            elif isinstance(fc, commands.FileRenameCommand):
                self.rename_old_paths.setdefault(cmd.id, set()).add(fc.old_path)
            elif isinstance(fc, commands.FileCopyCommand):
                self.copy_source_paths.setdefault(cmd.id, set()).add(fc.src_path)

        # Track the heads
        parents = self.reftracker.track_heads(cmd)

        # Track the parent counts
        parent_count = len(parents)
        try:
            self.parent_counts[parent_count] += 1
        except KeyError:
            self.parent_counts[parent_count] = 1
            if parent_count > self.max_parent_count:
                self.max_parent_count = parent_count

        # Remember the merges
        if cmd.merges:
            #self.merges.setdefault(cmd.ref, set()).update(cmd.merges)
            for merge in cmd.merges:
                if merge in self.merges:
                    self.merges[merge] += 1
                else:
                    self.merges[merge] = 1