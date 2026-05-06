def get_cls(project_name, project_data):
    """
    gets class from name and data, sets base level attrs
    defaults to facsimile.base.Facsimile
    """
    if project_name:
        cls = getattr(facsimile.base, project_data.get('class', 'Facsimile'))
        cls.name = project_name
    else:
        cls = facsimile.base.Facsimile
    return cls