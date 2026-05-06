def list(self, request, *args, **kwargs):
        """
        To get a list of customers, run GET against */api/customers/* as authenticated user. Note that a user can
        only see connected customers:

        - customers that the user owns
        - customers that have a project where user has a role

        Staff also can filter customers by user UUID, for example /api/customers/?user_uuid=<UUID>

        Staff also can filter customers by exists accounting_start_date, for example:

        The first category:
        /api/customers/?accounting_is_running=True
            has accounting_start_date empty (i.e. accounting starts at once)
            has accounting_start_date in the past (i.e. has already started).

        Those that are not in the first:
        /api/customers/?accounting_is_running=False # exists accounting_start_date

        """
        return super(CustomerViewSet, self).list(request, *args, **kwargs)