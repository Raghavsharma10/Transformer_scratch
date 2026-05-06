def next(self):
        '''
        Never return StopIteration
        '''
        if self.started is False:

            self.started = True
            now_ = datetime.now()
            if self.hour:
                # Fixed hour in a day
                # Next run will be the next day
                scheduled = now_.replace(hour=self.hour, minute=self.minute, second=self.second, microsecond=0)
                if scheduled == now_:
                    return timedelta(seconds=0)
                elif scheduled < now_:
                    # Scheduled time is passed
                    return scheduled.replace(day=now_.day + 1) - now_
            else:
                # Every hour in a day
                # Next run will be the next hour
                scheduled = now_.replace(minute=self.minute, second=self.second, microsecond=0)
                if scheduled == now_:
                    return timedelta(seconds=0)
                elif scheduled < now_:
                    # Scheduled time is passed
                    return scheduled.replace(hour=now_.hour + 1) - now_
            return scheduled - now_
        else:
            if self.hour:
                return timedelta(days=1)  # next day
            return timedelta(hours=1)