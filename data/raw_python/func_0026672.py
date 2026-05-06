def merge_duplicates(self):
        """Merge and remove duplicate entries.

        Compares each entry ('name') in `stubs` to all later entries to check
        for duplicates in name or alias.  If a duplicate is found, they are
        merged and written to file.
        """
        if len(self.entries) == 0:
            self.log.error("WARNING: `entries` is empty, loading stubs")
            if self.args.update:
                self.log.warning(
                    "No sources changed, entry files unchanged in update."
                    "  Skipping merge.")
                return
            self.entries = self.load_stubs()

        task_str = self.get_current_task_str()

        keys = list(sorted(self.entries.keys()))
        n1 = 0
        mainpbar = tqdm(total=len(keys), desc=task_str)
        while n1 < len(keys):
            name1 = keys[n1]
            if name1 not in self.entries:
                self.log.info("Entry for {} not found, likely already "
                              "deleted in merging process.".format(name1))
                n1 = n1 + 1
                mainpbar.update(1)
                continue
            allnames1 = set(self.entries[name1].get_aliases() + self.entries[
                name1].extra_aliases())

            # Search all later names
            for name2 in keys[n1 + 1:]:
                if name1 == name2:
                    continue
                if name1 not in self.entries:
                    self.log.info("Entry for {} not found, likely already "
                                  "deleted in merging process.".format(name1))
                    continue
                if name2 not in self.entries:
                    self.log.info("Entry for {} not found, likely already "
                                  "deleted in merging process.".format(name2))
                    continue

                allnames2 = set(self.entries[name2].get_aliases() +
                                self.entries[name2].extra_aliases())

                # If there are any common names or aliases, merge
                if len(allnames1 & allnames2):
                    self.log.warning("Found two entries with common aliases "
                                     "('{}' and '{}'), merging.".format(name1,
                                                                        name2))

                    load1 = self.proto.init_from_file(self, name=name1)
                    load2 = self.proto.init_from_file(self, name=name2)
                    if load1 is not None and load2 is not None:
                        # Delete old files
                        self._delete_entry_file(entry=load1)
                        self._delete_entry_file(entry=load2)
                        self.entries[name1] = load1
                        self.entries[name2] = load2
                        priority1 = 0
                        priority2 = 0
                        for an in allnames1:
                            if an.startswith(self.entries[name1]
                                             .priority_prefixes()):
                                priority1 += 1
                        for an in allnames2:
                            if an.startswith(self.entries[name2]
                                             .priority_prefixes()):
                                priority2 += 1

                        if priority1 > priority2:
                            self.copy_to_entry_in_catalog(name2, name1)
                            keys.append(name1)
                            del self.entries[name2]
                        else:
                            self.copy_to_entry_in_catalog(name1, name2)
                            keys.append(name2)
                            del self.entries[name1]
                    else:
                        self.log.warning('Duplicate already deleted')

                    # if len(self.entries) != 1:
                    #     self.log.error(
                    #         "WARNING: len(entries) = {}, expected 1.  "
                    #         "Still journaling...".format(len(self.entries)))
                    self.journal_entries()

            if self.args.travis and n1 > self.TRAVIS_QUERY_LIMIT:
                break
            n1 = n1 + 1
            mainpbar.update(1)
        mainpbar.close()