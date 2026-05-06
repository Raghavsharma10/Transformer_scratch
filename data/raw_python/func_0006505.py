def launch_background_job(self, job, on_error=None, on_success=None):
        """Launch the callable job in background thread.
        Succes or failure are controlled by on_error and on_success
        """
        if not self.main.mode_online:
            self.sortie_erreur_GUI(
                "Local mode activated. Can't run background task !")
            self.reset()
            return

        on_error = on_error or self.sortie_erreur_GUI
        on_success = on_success or self.sortie_standard_GUI

        def thread_end(r):
            on_success(r)
            self.update()

        def thread_error(r):
            on_error(r)
            self.reset()

        logging.info(
            f"Launching background task from interface {self.__class__.__name__} ...")
        th = threads.worker(job, thread_error, thread_end)
        self._add_thread(th)