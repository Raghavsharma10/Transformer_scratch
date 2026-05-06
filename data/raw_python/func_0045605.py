def _worker_method(self):
        """
        The body of the multi-get worker. Loops until
        :meth:`_should_quit` returns ``True``, taking tasks off the
        input queue, fetching the object, and putting them on the
        output queue.
        """
        while not self._should_quit():
            try:
                task = self._inq.get(block=True, timeout=0.25)
            except TypeError:
                if self._should_quit():
                    break
                else:
                    raise
            except Empty:
                continue

            try:
                # If data is found in cache this data is used.
                # Else, data is taken from riak and set to cache.
                if settings.ENABLE_CACHING:
                    obj_data = cache.get(task.key)
                    if obj_data:
                        task.outq.put((task.key, json.loads(obj_data)))
                        raise FoundInCache()

                btype = task.client.bucket_type(task.bucket_type)
                obj = btype.bucket(task.bucket).get(task.key, **task.options)

                if not obj.exists:
                    raise NotFound()

                if settings.ENABLE_CACHING:
                    cache.set(task.key, json.dumps(obj.data), settings.CACHE_EXPIRE_DURATION)

                task.outq.put((task.key, obj.data))

            except KeyboardInterrupt:
                raise
            except FoundInCache or NotFound:
                pass
            except Exception as err:
                errdata = (task.bucket_type, task.bucket, task.key, err)
                task.outq.put(errdata)
            finally:
                self._inq.task_done()