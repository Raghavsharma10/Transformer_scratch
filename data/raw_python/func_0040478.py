def print_fields(fields, sort_by_date=False, sort_by_open_projects=False):
    """
    Print a list of available fields and works
    sort_by_date : boolean whether we print works by their due date
    """
    if (not sort_by_date) and (not sort_by_open_projects):
        for (_, name, works) in fields:
            print(name)
            for work in works:
                print('- '+str(work))
    else:
        works = all_works
        # Sort works by due_date
        if sort_by_date:
            works.sort(key=lambda x: (not x.is_open, x.due_date), reverse=True)
        for work in works:
            if sort_by_open_projects:
                if not work.is_open:
                    continue
            # This is ugly, but there is no way to know the field name of a work without searching for it, at the moment
            field_name = [name for id, name, _ in fields if id == work.field][0]
            print(field_name)
            print('- '+str(work))