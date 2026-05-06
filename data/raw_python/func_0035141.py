def addStep(self, callback, *args, **kwargs):
    '''
    Add rollback step with optional arguments. If a rollback is
    triggered, each step is called in LIFO order.
    '''
    self.steps.append((callback, args, kwargs))