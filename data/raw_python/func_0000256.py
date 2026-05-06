def commit_manually(using=None):
    """
    Decorator that activates manual transaction control. It just disables
    automatic transaction control and doesn't do any commit/rollback of its
    own -- it's up to the user to call the commit and rollback functions
    themselves.
    """
    def entering(using):
        enter_transaction_management(using=using)

    def exiting(exc_value, using):
        leave_transaction_management(using=using)

    return _transaction_func(entering, exiting, using)