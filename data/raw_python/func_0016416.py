def initial_bill_date(self):
        '''
        An estimated initial bill date for an account created today,
        based on available plan info.
        '''
        time_to_start = None
        
        if self.initial_bill_count_unit == 'months':
            time_to_start = relativedelta(months=self.initial_bill_count)
        else:
            time_to_start = relativedelta(days=self.initial_bill_count)
        
        initial_bill_date = datetime.utcnow().date() + time_to_start
        
        return initial_bill_date