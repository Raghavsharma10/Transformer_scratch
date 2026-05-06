def indicator(self):
        """Produce the spinner."""

        while self.run:
            try:
                size = self.work_q.qsize()
            except Exception:
                note = 'Please wait '
            else:
                note = 'Number of Jobs in Queue = %s ' % size

            if self.msg:
                note = '%s %s' % (note, self.msg)

            for item in ['|', '/', '-', '\\']:
                sys.stdout.write('\rProcessing - [ %s ] - %s ' % (item, note))
                sys.stdout.flush()
                time.sleep(.1)
                self.run = self.run