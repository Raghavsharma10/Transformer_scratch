def schedule(self, name, timer, func, *args, **kwargs):
        '''
        ts = Scheduler('my_task')
        ts.schedule(every(seconds=10), handle_message, "Every 10 seconds")
        ts.schedule(every(seconds=30), fetch_url, url="http://yahoo.com", section="stock_ticker")
        ts.run_forever()
        '''
        self.tasks.append(Task(name, func, timer, *args, **kwargs))
        self.active[name] = []  # list of greenlets
        self.waiting[name] = []