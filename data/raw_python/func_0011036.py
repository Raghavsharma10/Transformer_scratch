def threadsafe_event_trigger(self, event_type):
        """Returns a callback to creates events, interrupting current event requests.

        Returned callback function will create an event of type event_type
        which will interrupt an event request if one
        is concurrently occuring, otherwise adding the event to a queue
        that will be checked on the next event request."""
        readfd, writefd = os.pipe()
        self.readers.append(readfd)

        def callback(**kwargs):
            self.queued_interrupting_events.append(event_type(**kwargs))  #TODO use a threadsafe queue for this
            logger.warning('added event to events list %r', self.queued_interrupting_events)
            os.write(writefd, b'interrupting event!')
        return callback