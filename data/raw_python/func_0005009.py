def inverse(self, vector, duration=None):
        '''Inverse vector transformer'''

        ann = jams.Annotation(namespace=self.namespace, duration=duration)

        if duration is None:
            duration = 0
        ann.append(time=0, duration=duration, value=vector)

        return ann