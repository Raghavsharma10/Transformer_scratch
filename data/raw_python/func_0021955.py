def concat(self, *args, **kwargs):
        """
        :type args: FormattedText
        :type kwargs: FormattedText
        """
        for arg in args:
            assert self.formatted_text._is_compatible(arg), "Cannot concat text with different modes"
            self.format_args.append(arg.text)
        for kwarg in kwargs:
            value = kwargs[kwarg]
            assert self.formatted_text._is_compatible(value), "Cannot concat text with different modes"
            self.format_kwargs[kwarg] = value.text
        return self