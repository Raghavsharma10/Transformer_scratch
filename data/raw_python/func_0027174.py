def change_customer_nc_users_quota(sender, structure, user, role, signal, **kwargs):
    """ Modify nc_user_count quota usage on structure role grant or revoke """
    assert signal in (signals.structure_role_granted, signals.structure_role_revoked), \
        'Handler "change_customer_nc_users_quota" has to be used only with structure_role signals'
    assert sender in (Customer, Project), \
        'Handler "change_customer_nc_users_quota" works only with Project and Customer models'

    if sender == Customer:
        customer = structure
    elif sender == Project:
        customer = structure.customer

    customer_users = customer.get_users()
    customer.set_quota_usage(Customer.Quotas.nc_user_count, customer_users.count())