def accountable_date(self):
        '''Accountable date of transaction, localized as America/Santiago
        '''
        fecha_transaccion = self.data['TBK_FECHA_CONTABLE']
        m = int(fecha_transaccion[:2])
        d = int(fecha_transaccion[2:])
        santiago = pytz.timezone('America/Santiago')
        today = santiago.localize(datetime.datetime.today())
        year = today.year
        if self.paid_at.month == 12 and m == 1:
            year += 1
        santiago_dt = santiago.localize(datetime.datetime(year, m, d))
        return santiago_dt