def transmit_agnocomplete_context(self):
        """
        We'll reset the current queryset only if the user is set.
        """
        user = super(AgnocompleteContextQuerysetMixin, self) \
            .transmit_agnocomplete_context()
        if user:
            self.queryset = self.agnocomplete.get_queryset()
        return user