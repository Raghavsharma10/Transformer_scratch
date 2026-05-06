def _request_bulk(self, urls: List[str]) -> List:
        """Batch the requests going out."""
        if not urls:
            raise Exception("No results were found")
        session: FuturesSession = FuturesSession(max_workers=len(urls))
        self.log.info("Bulk requesting: %d" % len(urls))
        futures = [session.get(u, headers=gen_headers(), timeout=3) for u in urls]
        done, incomplete = wait(futures)
        results: List = list()
        for response in done:
            try:
                results.append(response.result())
            except Exception as err:
                self.log.warn("Failed result: %s" % err)
        return results