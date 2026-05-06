def pages(self, page_from, page_to):
        """
        Yield torrents in range from page_from to page_to
        """
        if not all([page_from < self.url.max_page, page_from > 0,
                    page_to <= self.url.max_page, page_to > page_from]):
            raise IndexError("Invalid page numbers")

        size = (page_to + 1) - page_from
        threads = ret = []
        page_list = range(page_from, page_to+1)

        locks = [threading.Lock() for i in range(size)]

        for lock in locks[1:]:
            lock.acquire()

        def t_function(pos):
            """
            Thread function that fetch page for list of torrents
            """
            res = self.page(page_list[pos]).list()
            locks[pos].acquire()
            ret.extend(res)
            if pos != size-1:
                locks[pos+1].release()

        threads = [threading.Thread(target=t_function, args=(i,))
                   for i in range(size)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        for torrent in ret:
            yield torrent