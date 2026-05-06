def same_page_choosen(form, field):
    """Check that we are not trying to assign list page itself as a child."""
    if form._obj is not None:
        if field.data.id == form._obj.list_id:
            raise ValidationError(
                _('You cannot assign list page itself as a child.'))