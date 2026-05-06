def bulk_upsert(self, docs, namespace, timestamp):
        """Update or insert multiple documents into Solr

        docs may be any iterable
        """
        if self.auto_commit_interval is not None:
            add_kwargs = {
                "commit": (self.auto_commit_interval == 0),
                "commitWithin": str(self.auto_commit_interval)
            }
        else:
            add_kwargs = {"commit": False}

        cleaned = (self._clean_doc(d, namespace, timestamp) for d in docs)
        if self.chunk_size > 0:
            batch = list(next(cleaned) for i in range(self.chunk_size))
            while batch:
                self.solr.add(batch, **add_kwargs)
                batch = list(next(cleaned)
                             for i in range(self.chunk_size))
        else:
            self.solr.add(cleaned, **add_kwargs)