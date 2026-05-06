def journal_entries(self,
                        clear=True,
                        gz=False,
                        bury=False,
                        write_stubs=False,
                        final=False):
        """Write all entries in `entries` to files, and clear.  Depending on
        arguments and `tasks`.

        Iterates over all elements of `entries`, saving (possibly 'burying')
        and deleting.
        -   If ``clear == True``, then each element of `entries` is deleted,
            and a `stubs` entry is added
        """

        # if (self.current_task.priority >= 0 and
        #        self.current_task.priority < self.min_journal_priority):
        #    return

        # Write it all out!
        # NOTE: this needs to use a `list` wrapper to allow modification of
        # dict
        for name in list(self.entries.keys()):
            if self.args.write_entries:
                # If this is a stub and we aren't writing stubs, skip
                if self.entries[name]._stub and not write_stubs:
                    continue

                # Bury non-SN entries here if only claimed type is non-SN type,
                # or if primary name starts with a non-SN prefix.
                bury_entry = False
                save_entry = True
                if bury:
                    (bury_entry, save_entry) = self.should_bury(name)

                if save_entry:
                    save_name = self.entries[name].save(
                        bury=bury_entry, final=final)
                    self.log.info(
                        "Saved {} to '{}'.".format(name.ljust(20), save_name))
                    if (gz and os.path.getsize(save_name) >
                            self.COMPRESS_ABOVE_FILESIZE):
                        save_name = compress_gz(save_name)
                        self.log.debug(
                            "Compressed '{}' to '{}'".format(name, save_name))
                        # FIX: use subprocess
                        outdir, filename = os.path.split(save_name)
                        filename = filename.split('.')[0]
                        os.system('cd ' + outdir + '; git rm --cached ' +
                                  filename + '.json; git add -f ' + filename +
                                  '.json.gz; cd ' + self.PATHS.PATH_BASE)

            if clear:
                self.entries[name] = self.entries[name].get_stub()
                self.log.debug("Entry for '{}' converted to stub".format(name))

        return