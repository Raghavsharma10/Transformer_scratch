def _register_converter(cls, conv_func, conv_type):
        '''Triggered by the @converter_function decorator'''
        cls.converters.append(ConverterFunctionInfo(conv_func, conv_type, len(cls.converters)))
        cls._sort_converters()