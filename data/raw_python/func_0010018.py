def start(self):
        """
        Start standing by.  A periodic command like 'current_url' will be sent to the 
        webdriver instance to prevent it from timing out.

        """
        self._end_time = datetime.now() + timedelta(seconds=self._max_time)
        self._thread = Thread(target=lambda: self.__stand_by_loop())
        self._keep_running = True
        self._thread.start()
        return self