def re_enqueue(self, item):
        """Re-enqueue till reach max retries."""
        if 'retries' in item:
            retries = item['retries']
            if retries >= self.MAX_RETRIES:
                log.warn("Failed to execute {} after {} retries, give it "
                         " up.".format(item['method'], retries))
            else:
                retries += 1
                item['retries'] = retries
                self._q.put_nowait(item)
        else:
            item['retries'] = 1
            self._q.put_nowait(item)