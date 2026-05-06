def numericshape(self):
        """Shape of the array of temporary values required for the numerical
        solver actually being selected."""
        try:
            numericshape = [self.subseqs.seqs.model.numconsts.nmb_stages]
        except AttributeError:
            objecttools.augment_excmessage(
                'The `numericshape` of a sequence like `%s` depends on the '
                'configuration of the actual integration algorithm.  '
                'While trying to query the required configuration data '
                '`nmb_stages` of the model associated with element `%s`'
                % (self.name, objecttools.devicename(self)))
        # noinspection PyUnboundLocalVariable
        numericshape.extend(self.shape)
        return tuple(numericshape)