def set_text(self, input_text, *args, **selectors):
        """
        Set *input_text* to the UI object with *selectors* 
        """
        self.device(**selectors).set_text(input_text)