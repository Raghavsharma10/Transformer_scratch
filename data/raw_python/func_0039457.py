def _get_pages(self, locations, req):
        """Yields (page, page_url) from the given locations, skipping
        locations that have errors, and adding download/homepage links"""
        pending_queue = Queue()
        for location in locations:
            pending_queue.put(location)
        done = []
        seen = set()
        threads = []
        for i in range(min(10, len(locations))):
            t = threading.Thread(target=self._get_queued_page, args=(req, pending_queue, done, seen))
            t.setDaemon(True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        return done