def recompute_if_necessary(self, ui):
        """Recompute the data on a thread, if necessary.

        If the data has recently been computed, this call will be rescheduled for the future.

        If the data is currently being computed, it do nothing."""
        self.__initialize_cache()
        if self.__cached_value_dirty:
            with self.__is_recomputing_lock:
                is_recomputing = self.__is_recomputing
                self.__is_recomputing = True
            if is_recomputing:
                pass
            else:
                # the only way to get here is if we're not currently computing
                # this has the side effect of limiting the number of threads that
                # are sleeping.
                def recompute():
                    try:
                        if self.__recompute_thread_cancel.wait(0.01):  # helps tests run faster
                            return
                        minimum_time = 0.5
                        current_time = time.time()
                        if current_time < self.__cached_value_time + minimum_time:
                            if self.__recompute_thread_cancel.wait(self.__cached_value_time + minimum_time - current_time):
                                return
                        self.recompute_data(ui)
                    finally:
                        self.__is_recomputing = False
                        self.__recompute_thread = None
                with self.__is_recomputing_lock:
                    self.__recompute_thread = threading.Thread(target=recompute)
                    self.__recompute_thread.start()