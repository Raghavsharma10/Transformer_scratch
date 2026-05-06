def cmd_queue_peaks(self):
        """Generate a list of the requests peaks on the queue.

        A queue peak is defined by the biggest value on the backend queue
        on a series of log lines that are between log lines without being
        queued.

        .. warning::
          Allow to configure up to which peak can be ignored. Currently
          set to 1.
        """
        threshold = 1
        peaks = []
        current_peak = 0
        current_queue = 0

        current_span = 0
        first_on_queue = None

        for line in self._valid_lines:
            current_queue = line.queue_backend

            if current_queue > 0:
                current_span += 1

                if first_on_queue is None:
                    first_on_queue = line.accept_date

            if current_queue == 0 and current_peak > threshold:
                data = {
                    'peak': current_peak,
                    'span': current_span,
                    'first': first_on_queue,
                    'last': line.accept_date,
                }
                peaks.append(data)
                current_peak = 0
                current_span = 0
                first_on_queue = None

            if current_queue > current_peak:
                current_peak = current_queue

        # case of a series that does not end
        if current_queue > 0 and current_peak > threshold:
            data = {
                'peak': current_peak,
                'span': current_span,
                'first': first_on_queue,
                'last': line.accept_date,
            }
            peaks.append(data)

        return peaks