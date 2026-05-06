def get_queryset(self):
        """
        Making sure that a user can only edit his own images.

        Even when he forges the request URL.

        """
        queryset = super(UpdateImageView, self).get_queryset()
        queryset = queryset.filter(user=self.user)
        return queryset