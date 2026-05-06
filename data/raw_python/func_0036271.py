def _get_available_letters(field_name, queryset):
    """
    Makes a query to the database to return the first character of each
    value of the field and table passed in.

    Returns a set that represents the letters that exist in the database.
    """
    if django.VERSION[1] <= 4:
        result = queryset.values(field_name).annotate(
            fl=FirstLetter(field_name)
        ).values('fl').distinct()
        return set([res['fl'] for res in result if res['fl'] is not None])
    else:
        from django.db import connection
        qn = connection.ops.quote_name
        db_table = queryset.model._meta.db_table
        sql = "SELECT DISTINCT UPPER(SUBSTR(%s, 1, 1)) as letter FROM %s" % (qn(field_name), qn(db_table))
        cursor = connection.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall() or ()
        return set([row[0] for row in rows if row[0] is not None])