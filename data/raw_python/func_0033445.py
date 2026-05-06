def _save(self) -> None:
        """Save output to a directory."""
        self._log.info("Saving results to '%s'" % self.folder)
        path: str = self.folder + "/"
        for job in self.results:
            if job['domain'] in self.saved:
                continue
            job['start_time'] = str_datetime(job['start_time'])
            job['end_time'] = str_datetime(job['end_time'])
            jid: int = random.randint(100000, 999999)
            filename: str = "%s_%s_%d_job.json" % (self.project, job['domain'], jid)
            handle = open(path + filename, 'w')
            handle.write(json.dumps(job, indent=4))
            handle.close()

            filename = "%s_%s_%d_emails.txt" % (self.project, job['domain'], jid)
            handle = open(path + filename, 'w')
            for email in job['results']['emails']:
                handle.write(email + "\n")
            handle.close()
            self.saved.append(job['domain'])