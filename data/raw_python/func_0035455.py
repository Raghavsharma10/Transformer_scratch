def paid_at(self):
        '''Localized at America/Santiago datetime of ``TBK_FECHA_TRANSACCION``.
        '''
        fecha_transaccion = self.data['TBK_FECHA_TRANSACCION']
        hora_transaccion = self.data['TBK_HORA_TRANSACCION']
        m = int(fecha_transaccion[:2])
        d = int(fecha_transaccion[2:])
        h = int(hora_transaccion[:2])
        i = int(hora_transaccion[2:4])
        s = int(hora_transaccion[4:])

        santiago = pytz.timezone('America/Santiago')
        today = santiago.localize(datetime.datetime.today())
        santiago_dt = santiago.localize(
            datetime.datetime(today.year, m, d, h, i, s))

        return santiago_dt