def list(self, request, *args, **kwargs):
        """
        To get an actual value for object quotas limit and usage issue a **GET** request against */api/<objects>/*.

        To get all quotas visible to the user issue a **GET** request against */api/quotas/*
        """
        return super(QuotaViewSet, self).list(request, *args, **kwargs)