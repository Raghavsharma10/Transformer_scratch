def pages_counter(self, total: int, page_size: int = 100) -> int:
        """Simple helper to handle pagination. Returns the number of pages for a
        given number of results.

        :param int total: count of metadata in a search request
        :param int page_size: count of metadata to display in each page
        """
        if total <= page_size:
            count_pages = 1
        else:
            if (total % page_size) == 0:
                count_pages = total / page_size
            else:
                count_pages = (total / page_size) + 1
        # method ending
        return int(count_pages)