def most_seen_creators_by_works_card(work_kind=None, role_name=None, num=10):
    """
    Displays a card showing the Creators that are associated with the most Works.

    e.g.:

      {% most_seen_creators_by_works_card work_kind='movie' role_name='Director' num=5 %}
    """
    object_list = most_seen_creators_by_works(
                            work_kind=work_kind, role_name=role_name, num=num)

    object_list = chartify(object_list, 'num_works', cutoff=1)

    # Attempt to create a sensible card title...

    if role_name:
        # Yes, this pluralization is going to break at some point:
        creators_name = '{}s'.format(role_name.capitalize())
    else:
        creators_name = 'People/groups'

    if work_kind:
        works_name = Work.get_kind_name_plural(work_kind).lower()
    else:
        works_name = 'works'

    card_title = '{} with most {}'.format(creators_name, works_name)

    return {
        'card_title': card_title,
        'score_attr': 'num_works',
        'object_list': object_list,
    }