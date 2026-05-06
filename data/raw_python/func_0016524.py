def iterqueue(self, limit=None, infinite=False):
        """Infinite iterator yielding pending messages, by using
        synchronous direct access to the queue (``basic_get``).

        :meth:`iterqueue` is used where synchronous functionality is more
        important than performance. If you can, use :meth:`iterconsume`
        instead.

        :keyword limit: If set, the iterator stops when it has processed
            this number of messages in total.

        :keyword infinite: Don't raise :exc:`StopIteration` if there is no
            messages waiting, but return ``None`` instead. If infinite you
            obviously shouldn't consume the whole iterator at once without
            using a ``limit``.

        :raises StopIteration: If there is no messages waiting, and the
            iterator is not infinite.

        """
        for items_since_start in count():
            item = self.fetch()
            if (not infinite and item is None) or \
                    (limit and items_since_start >= limit):
                raise StopIteration
            yield item