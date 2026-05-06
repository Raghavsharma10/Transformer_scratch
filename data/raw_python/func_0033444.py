def _job_handler(self) -> bool:
        """Process the work items."""
        while True:
            try:
                task = self._unfullfilled.get_nowait()
            except queue.Empty:
                break
            else:
                self._log.debug("Job: %s" % str(task))
                engine = self._dyn_loader(task['engine'], task)
                task['start_time'] = now_time()
                results = engine.search()
                task['end_time'] = now_time()
                duration: str = str((task['end_time'] - task['start_time']).seconds)
                task['duration'] = duration
                task.update({'results': results})
                self._fulfilled.put(task)
        return True