def normalize_rust_function(self, function, line):
        """Normalizes a single rust frame with a function"""
        # Drop the prefix and return type if there is any
        function = drop_prefix_and_return_type(function)

        # Collapse types
        function = collapse(
            function,
            open_string='<',
            close_string='>',
            replacement='<T>',
            exceptions=(' as ',)
        )

        # Collapse arguments
        if self.collapse_arguments:
            function = collapse(
                function,
                open_string='(',
                close_string=')',
                replacement=''
            )

        if self.signatures_with_line_numbers_re.match(function):
            function = '{}:{}'.format(function, line)

        # Remove spaces before all stars, ampersands, and commas
        function = self.fixup_space.sub('', function)

        # Ensure a space after commas
        function = self.fixup_comma.sub(', ', function)

        # Remove rust-generated uniqueness hashes
        function = self.fixup_hash.sub('', function)

        return function