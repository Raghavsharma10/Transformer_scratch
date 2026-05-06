def _clear(self, wait):
        """
        clear outs the all content of current bucket
        only for development purposes
        """
        i = 0
        t1 = time.time()
        for k in self.bucket.get_keys():
            i += 1
            self.bucket.get(k).delete()
        print("\nDELETION TOOK: %s" % round(time.time() - t1, 2))
        if wait:
            while self._model_class.objects.count():
                time.sleep(0.3)
        return i