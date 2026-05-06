def marking_selector_syntax(instance):
    """Ensure selectors in granular markings refer to items which are actually
    present in the object.
    """
    if 'granular_markings' not in instance:
        return

    list_index_re = re.compile(r"\[(\d+)\]")
    for marking in instance['granular_markings']:
        if 'selectors' not in marking:
            continue

        selectors = marking['selectors']
        for selector in selectors:
            segments = selector.split('.')

            obj = instance
            prev_segmt = None
            for segmt in segments:
                index_match = list_index_re.match(segmt)
                if index_match:
                    try:
                        idx = int(index_match.group(1))
                        obj = obj[idx]
                    except IndexError:
                        yield JSONError("'%s' is not a valid selector because"
                                        " %s is not a valid index."
                                        % (selector, idx), instance['id'])
                    except KeyError:
                        yield JSONError("'%s' is not a valid selector because"
                                        " '%s' is not a list."
                                        % (selector, prev_segmt), instance['id'])
                else:
                    try:
                        obj = obj[segmt]
                    except KeyError as e:
                        yield JSONError("'%s' is not a valid selector because"
                                        " %s is not a property."
                                        % (selector, e), instance['id'])
                    except TypeError:
                        yield JSONError("'%s' is not a valid selector because"
                                        " '%s' is not a property."
                                        % (selector, segmt), instance['id'])
                prev_segmt = segmt