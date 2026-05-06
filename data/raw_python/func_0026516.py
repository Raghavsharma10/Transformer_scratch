def notify(self, event):
        """Notify a user"""

        self.log('Got a notification event!')

        self.log(event, pretty=True)
        self.log(event.__dict__)