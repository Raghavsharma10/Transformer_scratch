def update(self, pbar):
        'Updates the widget to show the ETA or total time when finished.'
        self.n_refresh += 1
        if pbar.currval == 0:
            return 'ETA:  --:--:--'
        elif pbar.finished:
            return 'Time: %s' % self.format_time(pbar.seconds_elapsed)
        else:
            elapsed = pbar.seconds_elapsed
            try:
                speed = pbar.currval / elapsed
                if self.speed_smooth is not None:
                    self.speed_smooth = (self.speed_smooth * (1 - self.SMOOTHING)) + (speed * self.SMOOTHING)
                else:
                    self.speed_smooth = speed
                eta = float(pbar.maxval) / self.speed_smooth - elapsed + 1 if float(pbar.maxval) / self.speed_smooth - elapsed + 1 > 0 else 0

                if float(pbar.currval) / pbar.maxval > 0.30 or self.n_refresh > 10:  # ETA only rather precise if > 30% is already finished or more than 10 times updated
                    return 'ETA:  %s' % self.format_time(eta)
                if self.old_eta is not None and self.old_eta < eta:  # do not show jumping ETA if non precise mode is active
                    return 'ETA: ~%s' % self.format_time(self.old_eta)
                else:
                    self.old_eta = eta
                    return 'ETA: ~%s' % self.format_time(eta)
            except ZeroDivisionError:
                speed = 0