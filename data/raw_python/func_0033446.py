def search(self, jobs: List[Dict[str, str]]) -> None:
        """Perform searches based on job orders."""
        if not isinstance(jobs, list):
            raise Exception("Jobs must be of type list.")
        self._log.info("Project: %s" % self.project)
        self._log.info("Processing jobs: %d", len(jobs))
        for _, job in enumerate(jobs):
            self._unfullfilled.put(job)

        for _ in range(self.PROCESSES):
            proc: Process = Process(target=self._job_handler)
            self._processes.append(proc)
            proc.start()

        for proc in self._processes:
            proc.join()

        while not self._fulfilled.empty():
            output: Dict = self._fulfilled.get()
            output.update({'project': self.project})
            self._processed.append(output['domain'])
            self.results.append(output)

            if output['greedy']:
                bonus_jobs: List = list()
                observed: List = list()
                for item in output['results']['emails']:
                    found: str = item.split('@')[1]
                    if found in self._processed or found in observed:
                        continue
                    observed.append(found)
                    base: Dict = dict()
                    base['limit'] = output['limit']
                    base['modifier'] = output['modifier']
                    base['engine'] = output['engine']
                    base['greedy'] = False
                    base['domain'] = found
                    bonus_jobs.append(base)

                if len(bonus_jobs) > 0:
                    self.search(bonus_jobs)

        self._log.info("All jobs processed")
        if self.output:
            self._save()