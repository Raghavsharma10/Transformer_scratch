def inverse(self, encoded, duration=None):
        '''Inverse static tag transformation'''

        ann = jams.Annotation(namespace=self.namespace, duration=duration)

        if np.isrealobj(encoded):
            detected = (encoded >= 0.5)
        else:
            detected = encoded

        for vd in self.encoder.inverse_transform(np.atleast_2d(detected))[0]:
            vid = np.flatnonzero(self.encoder.transform(np.atleast_2d(vd)))
            ann.append(time=0,
                       duration=duration,
                       value=vd,
                       confidence=encoded[vid])
        return ann