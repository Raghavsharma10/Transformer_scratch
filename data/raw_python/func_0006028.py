def parse_locator(locator):
        """
        Parses a valid selenium By and value from a locator;
        returns as a named tuple with properties 'By' and 'value'

        locator -- a valid element locator or css string
        """

        # handle backwards compatibility to support new Locator class
        if isinstance(locator, loc.Locator):
            locator = '{by}={locator}'.format(by=locator.by, locator=locator.locator)

        locator_tuple = namedtuple('Locator', 'By value')

        if locator.count('=') > 0 and locator.count('css=') < 1:
            by = locator[:locator.find('=')].replace('_', ' ')
            value = locator[locator.find('=')+1:]
            return locator_tuple(by, value)
        else:  # assume default is css selector
            value = locator[locator.find('=')+1:]
            return locator_tuple('css selector', value)