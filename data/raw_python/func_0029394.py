def output_results(cls, sprites):
        """Output whether or not each attribute was correctly initialized.

        Attributes that were not modified at all are considered to be properly
        initialized.

        """
        print(' '.join(cls.ATTRIBUTES))
        format_strs = ['{{{}!s:^{}}}'.format(x, len(x)) for x in
                       cls.ATTRIBUTES]
        print(' '.join(format_strs).format(**cls.attribute_result(sprites)))