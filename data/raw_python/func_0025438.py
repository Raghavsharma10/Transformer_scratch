def text_filter_changed(self, text):
        """
            Called to handle changes to the text filter.

            :param text: The text for the filter.
        """
        text = text.strip() if text else None

        if text is not None:
            self.__text_filter = ListModel.TextFilter("text_for_filter", text)
        else:
            self.__text_filter = None

        self.__update_filter()