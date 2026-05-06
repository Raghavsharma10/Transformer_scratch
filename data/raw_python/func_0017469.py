def log(self, name, val, **tags):
        """Log metric name with value val. You must include at least one tag as a kwarg"""
        global _last_timestamp, _last_metrics

        # do not allow .log after closing
        assert not self.done.is_set(), "worker thread has been closed"
        # check if valid metric name
        assert all(c in _valid_metric_chars for c in name), "invalid metric name " + name

        val = float(val)  #Duck type to float/int, if possible.
        if int(val) == val:
            val = int(val)

        if self.host_tag and 'host' not in tags:
            tags['host'] = self.host_tag

        # get timestamp from system time, unless it's supplied as a tag
        timestamp = int(tags.pop('timestamp', time.time()))

        assert not self.done.is_set(), "tsdb object has been closed"
        assert tags != {}, "Need at least one tag"

        tagvals = ' '.join(['%s=%s' % (k, v) for k, v in tags.items()])

        # OpenTSDB has major problems if you insert a data point with the same
        # metric, timestamp and tags. So we keep a temporary set of what points
        # we have sent for the last timestamp value. If we encounter a duplicate,
        # it is dropped.
        unique_str = "%s, %s, %s, %s, %s" % (name, timestamp, tagvals, self.host, self.port)
        if timestamp == _last_timestamp or _last_timestamp == None:
            if unique_str in _last_metrics:
                return  # discard duplicate metrics
            else:
                _last_metrics.add(unique_str)
        else:
            _last_timestamp = timestamp
            _last_metrics.clear()

        line = "put %s %d %s %s\n" % (name, timestamp, val, tagvals)

        try:
            self.q.put(line, False)
            self.queued += 1
        except queue.Full:
            print("potsdb - Warning: dropping oldest metric because Queue is full. Size: %s" % self.q.qsize(), file=sys.stderr)
            self.q.get()  #Drop the oldest metric to make room
            self.q.put(line, False)
        return line