def cmd_average_response_time(self):
        """Returns the average response time of all, non aborted, requests."""
        average = [
            line.time_wait_response
            for line in self._valid_lines
            if line.time_wait_response >= 0
        ]

        divisor = float(len(average))
        if divisor > 0:
            return sum(average) / float(len(average))
        return 0