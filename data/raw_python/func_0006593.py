def compose(self, text, minimal_clears=False, no_clears=False):
        '''
        Returns the sequence of combinations necessary to compose given text.

        If the text expression is not possible with the given layout an ComposeException is thrown.

        Iterate over the string, converting each character into a key sequence.
        Between each character, an empty combo is inserted to handle duplicate strings (and USB HID codes between characters)

        @param text: Input UTF-8 string
        @param minimal_clears: Set to True to minimize the number of code clears. False (default) includes a clear after every character.
        @param no_clears: Set to True to not add any code clears (useful for input sequences). False (default) to include code clears.

        @returns: Sequence of combinations needed to generate the given text string
        '''
        sequence = []
        clear = self.json_data['to_hid_keyboard']['0x00'] # No Event

        for char in text:
            # Make sure the composition element is available
            if char not in self.json_data['composition']:
                raise ComposeException("'{}' is not defined as a composition in the layout '{}'".format(char, self.name))

            # Lookup the sequence to handle this character
            lookup = self.json_data['composition'][char]

            # If using minimal clears, check to see if we need to re-use any codes
            # Only need to check the most recent addition with the first combo
            if sequence and set(tuple(lookup[0])) & set(tuple(sequence[-1])) and not no_clears:
                sequence.extend([[clear]])

            # Add to overall sequence
            sequence.extend(lookup)

            # Add empty combo for sequence splitting
            if not minimal_clears and not no_clears:
                # Blindly add a clear combo between characters
                sequence.extend([[clear]])

        # When using minimal clears, we still need to add a final clear
        if minimal_clears and not no_clears:
            sequence.extend([[clear]])

        return sequence