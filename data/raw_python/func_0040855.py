def rollback(self):
        """ Do journal rollback """

        # Close the journal for writing, if this is an automatic rollback following a crash,
        # the file descriptor will not be open, so don't need to do anything.
        if self.journal != None: self.journal.close()
        self.journal = None

        # Read the journal
        journ_list = []
        with open(self.j_file) as fle:
            for l in fle: journ_list.append(json.loads(l))

        journ_subtract = deque(reversed(journ_list))

        for j_itm in reversed(journ_list):
            try: self.do_action({'do' : j_itm}, False)
            except IOError: pass

            # As each item is completed remove it from the journal file, in case
            # something fails during the rollback we can pick up where it stopped.
            journ_subtract.popleft()
            with open(self.j_file, 'w') as f:
                for data in list(journ_subtract):
                    f.write(json.dumps(data) + "\n")
                f.flush()

        # Rollback is complete so delete the journal file
        os.remove(self.j_file)