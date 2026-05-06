def item_land_adapter(obj, request):
    """
    Adapter for rendering an item of
    :class: `pycountry.db.Data` to json.
    """
    return {
        'id': obj.alpha_2,
        'alpha2': obj.alpha_2,
        'alpha3': obj.alpha_3,
        'naam': _(obj.name)
    }