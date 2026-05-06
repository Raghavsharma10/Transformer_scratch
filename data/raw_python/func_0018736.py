def notify_all(self):
        """Notifies all waiting threads"""
        self.condition.acquire()
        # python 3.x
        try:
            self.condition.notify_all()
        except:
            self.condition.notifyAll()
        self.condition.release()