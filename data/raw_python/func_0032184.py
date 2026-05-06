def getReadSession(self):
        ''' return scopted session '''
        if self.ReadSession is None:
            self.ReadSession=scoped_session(sessionmaker(bind=self.engine))

        return self.ReadSession