def deselect_option(self, value=None, text=None, index=None):
        """
        De-selects an option by value, text, or index. You must name the parameter

        @type value:    str
        @param value:   the value of the option
        @type text:     str
        @param text:    the option's visible text
        @type index:    int
        @param index:   the zero-based index of the option

        @rtype:     WebElementWrapper
        @return:    self
        """
        def do_deselect():
            """
            Perform selection
            """
            return self.set_select('deselect', value, text, index)
        return self.execute_and_handle_webelement_exceptions(do_deselect, 'deselect option')