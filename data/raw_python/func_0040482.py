def add_server(self,address,port=default_port,password=None,speed=None,valid_times=None,invalid_times=None):
        '''
        :address:           remote address of server, or special string ``local`` to
                            run the command locally
        :valid_times:       times when this server is available, given as a list
                            of tuples of 2 strings of form "HH:MM" that define the
                            start and end times. Alternatively, a list of 7 lists can
                            be given to define times on a per-day-of-week basis
        E.g.,::
        
            [('4:30','14:30'),('17:00','23:00')]
            # or
            [
                [('4:30','14:30'),('17:00','23:00')],       # S
                [('4:30','14:30'),('17:00','23:00')],       # M
                [('4:30','14:30'),('17:00','23:00')],       # T
                [('4:30','14:30'),('17:00','23:00')],       # W
                [('4:30','14:30'),('17:00','23:00')],       # R
                [('4:30','14:30'),('17:00','23:00')],       # F
                [('4:30','14:30'),('17:00','23:00')]        # S
            ]
        
        :invalid_times:     uses the same format as ``valid_times`` but defines times
                            when the server should not be used
        '''
        for t in [valid_times,invalid_times]:
            if t:
                if not (self._is_list_of_tuples(t) or self._is_list_of_tuples(t,True)):
                    raise ValueError('valid_times and invalid_times must either be lists of strings or lists')
        self.servers.append({
            'address':address,
            'port':port,
            'password':password,
            'speed':speed,
            'valid_times':valid_times,
            'invalid_times':invalid_times
        })