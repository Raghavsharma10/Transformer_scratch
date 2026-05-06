def feed_amount(self, amount):
        '''Calling this function sets the form feed amount to the specified setting.
    
        Args:
            amount: the form feed setting you desire. Options are '1/8', '1/6', 'x/180', and 'x/60',
            with x being your own desired amount. X must be a minimum of 24 for 'x/180' and 8 for 'x/60'
        Returns:
            None
        Raises:
            None
        '''
        n = None
        if amount=='1/8':
            amount = '0'
        elif amount=='1/6':
            amount = '2'
        elif re.search('/180', amount):
            n = re.search(r"(\d+)/180", amount)
            n = n.group(1)
            amount = '3'
        elif re.search('/60', amount):
            n = re.search(r"(\d+)/60", amount)
            n = n.group(1)
            amount = 'A'
        if n:
            self.send(chr(27)+amount+n)
        else:
            self.send(chr(27)+amount)