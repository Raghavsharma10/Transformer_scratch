def n_day(date_string):
        """
        date_string string in format "(number|a) day(s) ago"
        """
        today = datetime.date.today()
        match = re.match(r'(\d{1,3}|a) days? ago', date_string)
        groups = match.groups()
        if groups:
            decrement = groups[0]
            if decrement == 'a':
                decrement = 1
            return today - datetime.timedelta(days=int(decrement))
        return None