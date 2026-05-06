def prepend_model(self, value, model):
        """
        Prepends model name if it is not already prepended.
        For example model is "Offer":

            key -> Offer.key
            -key -> -Offer.key
            Offer.key -> Offer.key
            -Offer.key -> -Offer.key
        """
        if '.' not in value:
            direction = ''
            if value.startswith('-'):
                value = value[1:]
                direction = '-'
            value = '%s%s.%s' % (direction, model, value)
        return value