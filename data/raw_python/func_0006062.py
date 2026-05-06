def set_select(self, select_or_deselect = 'select', value=None, text=None, index=None):
        """
        Private method used by select methods

        @type select_or_deselect:   str
        @param select_or_deselect:  Should I select or deselect the element
        @type value:                str
        @type value:                Value to be selected
        @type text:                 str
        @type text:                 Text to be selected
        @type index:                int
        @type index:                index to be selected

        @rtype:     WebElementWrapper
        @return:    Self
        """
        # TODO: raise exception if element is not select element

        if select_or_deselect is 'select':
            if value is not None:
                Select(self.element).select_by_value(value)
            elif text is not None:
                Select(self.element).select_by_visible_text(text)
            elif index is not None:
                Select(self.element).select_by_index(index)

        elif select_or_deselect is 'deselect':
            if value is not None:
                Select(self.element).deselect_by_value(value)
            elif text is not None:
                Select(self.element).deselect_by_visible_text(text)
            elif index is not None:
                Select(self.element).deselect_by_index(index)

        elif select_or_deselect is 'deselect all':
            Select(self.element).deselect_all()

        return self