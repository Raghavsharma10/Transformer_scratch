def clear_tables_for_loaddata(confirm=None):
    """
    Clears al tables in order to loaddata properly.
    :param string:
    :return:
    """
    from django.contrib.contenttypes.models import ContentType
    from django.contrib.auth.models import Permission
    from django.contrib.sites.models import Site

    if confirm != 'yes':
        print('Please enter "yes" to confirm that your want to clear ContentTypes, Sites, Permissions')
    else:
        Site.objects.all().delete()
        Permission.objects.all().delete()
        ContentType.objects.all().delete()