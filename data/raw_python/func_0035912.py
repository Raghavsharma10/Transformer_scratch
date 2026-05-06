def list(cls, session, first_name=None, last_name=None, email=None,
             modified_since=None):
        """List the customers.

        Customers can be filtered on any combination of first name, last name,
        email, and modifiedSince.

        Args:
            session (requests.sessions.Session): Authenticated session.
            first_name (str, optional): First name of customer.
            last_name (str, optional): Last name of customer.
            email (str, optional): Email address of customer.
            modified_since (datetime.datetime, optional): If modified after
                this date.

        Returns:
            RequestPaginator(output_type=helpscout.models.Customer): Customers
                iterator.
        """
        return super(Customers, cls).list(
            session,
            data=cls.__object__.get_non_empty_vals({
                'firstName': first_name,
                'lastName': last_name,
                'email': email,
                'modifiedSince': modified_since,
            })
        )