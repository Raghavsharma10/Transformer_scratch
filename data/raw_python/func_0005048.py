def empty(self, duration):
        '''Create an empty jams.Annotation for this task.

        This method should be overridden by derived classes.

        Parameters
        ----------
        duration : int >= 0
            Duration of the annotation
        '''
        return jams.Annotation(namespace=self.namespace, time=0, duration=0)