def get_age(dt):
        """Calculate delta between current time and datetime and return a human readable form of the delta object"""
        delta = datetime.now() - dt
        days = delta.days
        hours, rem = divmod(delta.seconds, 3600)
        minutes, seconds = divmod(rem, 60)

        if days:
            return '%dd %dh %dm' % (days, hours, minutes)
        else:
            return '%dh %dm %ds' % (hours, minutes, seconds)