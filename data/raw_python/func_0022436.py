def get_count(self, *args, **selectors):
        """
        Return the count of UI object with *selectors*

        Example:

        | ${count}              | Get Count           | text=Accessibility    | # Get the count of UI object text=Accessibility |
        | ${accessibility_text} | Get Object          | text=Accessibility    | # These two keywords combination                |
        | ${count}              | Get Count Of Object | ${accessibility_text} | # do the same thing.                            |

        """
        obj = self.get_object(**selectors)
        return self.get_count_of_object(obj)