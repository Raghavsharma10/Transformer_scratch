def items(self, query=None, **kwargs):
        """
        Return the items to be sent to the client
        """
        # Cut this, we don't need no empty query
        if not query:
            self.__final_queryset = self.get_model().objects.none()
            return self.serialize(self.__final_queryset)
        # Query is too short, no item
        if len(query) < self.get_query_size_min():
            self.__final_queryset = self.get_model().objects.none()
            return self.serialize(self.__final_queryset)

        if self.requires_authentication:
            if not self.user:
                raise AuthenticationRequiredAgnocompleteException(
                    "Authentication is required to use this autocomplete"
                )
            if not self.user.is_authenticated:
                raise AuthenticationRequiredAgnocompleteException(
                    "Authentication is required to use this autocomplete"
                )

        qs = self.build_filtered_queryset(query, **kwargs)
        # The final queryset is the paginated queryset
        self.__final_queryset = qs
        return self.serialize(qs)