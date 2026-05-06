def normalize_cpp_function(self, function, line):
        """Normalizes a single cpp frame with a function"""
        # Drop member function cv/ref qualifiers like const, const&, &, and &&
        for ref in ('const', 'const&', '&&', '&'):
            if function.endswith(ref):
                function = function[:-len(ref)].strip()

        # Drop the prefix and return type if there is any if it's not operator
        # overloading--operator overloading syntax doesn't have the things
        # we're dropping here and can look curious, so don't try
        if '::operator' not in function:
            function = drop_prefix_and_return_type(function)

        # Collapse types
        function = collapse(
            function,
            open_string='<',
            close_string='>',
            replacement='<T>',
            exceptions=('name omitted', 'IPC::ParamTraits')
        )

        # Collapse arguments
        if self.collapse_arguments:
            function = collapse(
                function,
                open_string='(',
                close_string=')',
                replacement='',
                exceptions=('anonymous namespace', 'operator')
            )

        # Remove PGO cold block labels like "[clone .cold.222]". bug #1397926
        if 'clone .cold' in function:
            function = collapse(
                function,
                open_string='[',
                close_string=']',
                replacement=''
            )

        if self.signatures_with_line_numbers_re.match(function):
            function = '{}:{}'.format(function, line)

        # Remove spaces before all stars, ampersands, and commas
        function = self.fixup_space.sub('', function)

        # Ensure a space after commas
        function = self.fixup_comma.sub(', ', function)

        return function