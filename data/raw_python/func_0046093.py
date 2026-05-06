def _convert_duration_to_hhmmss(self, duration):
        """stub"""
        time_secs = duration.seconds
        min_, sec = divmod(time_secs, 60)
        hour, min_ = divmod(min_, 60)
        results = {
            'hours': hour,
            'minutes': min_,
            'seconds': sec
        }

        return results