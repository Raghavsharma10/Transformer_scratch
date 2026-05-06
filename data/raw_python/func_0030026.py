def expanded_transform(self):
        """Expands the transform string into segments """

        segments = self._expand_transform(self.transform)

        if segments:

            segments[0]['datatype'] = self.valuetype_class

            for s in segments:
                s['column'] = self

        else:

            segments = [self.make_xform_seg(datatype=self.valuetype_class, column=self)]

        # If we want to add the find datatype cast to a transform.
        #segments.append(self.make_xform_seg(transforms=["cast_"+self.datatype], column=self))

        return segments