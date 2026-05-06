def __update_filter(self):
        """
            Create a combined filter. Set the resulting filter into the document controller.
        """
        filters = list()
        if self.__date_filter:
            filters.append(self.__date_filter)
        if self.__text_filter:
            filters.append(self.__text_filter)
        self.document_controller.display_filter = ListModel.AndFilter(filters)