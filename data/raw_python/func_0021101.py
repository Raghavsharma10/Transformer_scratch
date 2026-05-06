def get_kind_name_plural(kind):
        "e.g. 'Gigs' or 'Movies'."
        if kind in ['comedy', 'cinema', 'dance', 'theatre']:
            return kind.title()
        elif kind == 'museum':
            return 'Galleries/Museums'
        else:
            return '{}s'.format(Event.get_kind_name(kind))