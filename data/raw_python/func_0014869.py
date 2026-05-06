def prepare_sort_params(params, request, sort_key='sort', revers_sort=None, except_params=None):
    """
        Prepare sort params. Add revers '-' if need.
        Params:
            params - list of sort parameters
            request
            sort_key
            revers_sort - list or set with keys that need reverse default sort direction
            except_params - GET-params that will be ignored
        Example:
            view:
                c['sort_params'] = prepare_sort_params(
                    ('order__lab_number', 'order__client__lname', 'organization', 'city', 'street', ),
                    request,
                )
            template:
                   <th><a href="{{ sort_params.order__lab_number.url }}">Лабораторный номер</a></th>
               or
                    {% load djutils %}
                    ...
                    {% sort_th 'order__lab_number' 'Лабораторный номер' %}


    """
    current_param, current_reversed = sort_key_process(request, sort_key)

    except_params = except_params or []
    except_params.append(sort_key)

    base_url = url_params(request, except_params=except_params, as_is=True)

    sort_params = {}
    revers_sort = revers_sort or set()
    url_connector = '?' if request.get_full_path() == request.path else "&"
    for p in params:
        sort_params[p] = {}
        if current_param and p == current_param:
            prefix = '' if current_reversed else '-'
            sort_params[p]['url'] = base_url + "%s%s=%s" % (url_connector, sort_key, prefix + current_param)
            sort_params[p]['is_reversed'] = current_reversed
            sort_params[p]['is_current'] = True
        else:
            default_direction = '-' if p in revers_sort else ''
            sort_params[p]['url'] = base_url + "%s%s=%s%s" % (url_connector, sort_key, default_direction, p)
            sort_params[p]['is_reversed'] = False
            sort_params[p]['is_current'] = False

    return sort_params