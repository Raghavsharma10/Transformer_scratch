def _setup_direct_converter(self, converter):
        '''
        Given a converter, set up the direct_output routes for conversions,
        which is used for transcoding between similar datatypes.
        '''
        inputs = (
            converter.direct_inputs
            if hasattr(converter, 'direct_inputs')
            else converter.inputs
        )
        for in_ in inputs:
            for out in converter.direct_outputs:
                self.direct_converters[(in_, out)] = converter