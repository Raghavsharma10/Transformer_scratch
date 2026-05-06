def linked_form(viewset, form_id=None, link=None, link_id=None, method=None):
    """
    When having foreign key or m2m relationships between models A and B (B has foreign key to A named parent),
    we want to have a form that sits on A's viewset but creates/edits B and sets it relationship to A
    automatically.

    In order to do so, define linked_forms on A's viewset containing a call to linked_form as follows:

    @linked_forms()
    class AViewSet(AngularFormMixin, ...):
        linked_forms = {
            'new-b': linked_form(BViewSet, link='parent')
        }

    Then, there will be a form definition on <aviewset>/pk/forms/new-b, with POST/PATCH operations pointing
    to an automatically created endpoint <aviewset>/pk/linked-endpoint/new-b and detail-route named "new_b"

    :param viewset:     the foreign viewset
    :param form_id:     id of the form on the foreign viewset. If unset, use the default form
    :param link:        either a field name on the foreign viewset or a callable that will get (foreign_instance, this_instance)
    :return:            an internal definition of a linked form
    """
    return {
        'viewset' : viewset,
        'form_id' : form_id,
        'link'    : link,
        'link_id' : link_id,
        'method'  : method
    }