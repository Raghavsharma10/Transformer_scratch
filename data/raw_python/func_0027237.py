def retrieve(self, request, *args, **kwargs):
        """
        To remove a link, issue **DELETE** to URL of the corresponding connection as stuff user or customer owner.
        """
        return super(BaseServiceProjectLinkViewSet, self).retrieve(request, *args, **kwargs)