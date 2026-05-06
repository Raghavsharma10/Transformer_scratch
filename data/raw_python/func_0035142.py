def doRollback(self):
    '''
    Call each rollback step in LIFO order.
    '''
    while self.steps:
      callback, args, kwargs = self.steps.pop()
      callback(*args, **kwargs)