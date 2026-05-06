def stop(self):
        """Stops services"""
        logging.debug('Stopping drone, hang on.')
        if self.drone is not None:
            self.drone_greenlet.unlink(self.on_exception)
            self.drone.stop()
            self.drone_greenlet.kill()
            self.drone = None
        # just some time for the drone to powerdown to be nice.
        gevent.sleep(2)
        if self.drone_greenlet is not None:
            self.drone_greenlet.kill(timeout=5)