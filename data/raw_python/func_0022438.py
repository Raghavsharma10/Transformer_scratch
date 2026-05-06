def call(self, obj, method, *args, **selectors):
        """
        This keyword can use object method from original python uiautomator

        See more details from https://github.com/xiaocong/uiautomator

        Example:

        | ${accessibility_text} | Get Object            | text=Accessibility | # Get the UI object                        |
        | Call                  | ${accessibility_text} | click              | # Call the method of the UI object 'click' |
        """
        func = getattr(obj, method)
        return func(**selectors)