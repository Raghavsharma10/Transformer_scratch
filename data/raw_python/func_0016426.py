def _determine_weights(self, other, settings):
        """
        Return weights of name components based on whether or not they were
        omitted
        """

        # TODO: Reduce weight for matches by prefix or initials

        first_is_used = settings['first']['required'] or \
            self.first and other.first
        first_weight = settings['first']['weight'] if first_is_used else 0

        middle_is_used = settings['middle']['required'] or \
            self.middle and other.middle
        middle_weight = settings['middle']['weight'] if middle_is_used else 0

        last_is_used = settings['last']['required'] or \
            self.last and other.last
        last_weight = settings['last']['weight'] if last_is_used else 0

        return first_weight, middle_weight, last_weight