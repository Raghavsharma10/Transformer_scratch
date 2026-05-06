def process_response(self, request, response):
        """Commits and leaves transaction management."""
        if tldap.transaction.is_managed():
            tldap.transaction.commit()
            tldap.transaction.leave_transaction_management()
        return response