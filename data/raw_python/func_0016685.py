def get_queryset(self):
        """
        Making sure that a user can only delete his own images.

        Even when he forges the request URL.

        """
        queryset = super(DeleteImageView, self).get_queryset()
        queryset = queryset.filter(user=self.user)
        return queryset