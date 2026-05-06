def qs_alphabet_filter(parser, token):
    """
    The parser/tokenizer for the queryset alphabet filter.

    {% qs_alphabet_filter <queryset> <field name> [<template name>] [strip_params=comma,delim,list] %}

    {% qs_alphabet_filter objects lastname myapp/template.html %}

    The template name is optional and uses alphafilter/alphabet.html if not
    specified
    """
    bits = token.split_contents()
    if len(bits) == 3:
        return AlphabetFilterNode(bits[1], bits[2])
    elif len(bits) == 4:
        if "=" in bits[3]:
            key, val = bits[3].split('=')
            return AlphabetFilterNode(bits[1], bits[2], strip_params=val)
        else:
            return AlphabetFilterNode(bits[1], bits[2], template_name=bits[3])
    elif len(bits) == 5:
        key, val = bits[4].split('=')
        return AlphabetFilterNode(bits[1], bits[2], bits[3], bits[4])
    else:
        raise TemplateSyntaxError("%s is called with a queryset and field "
                                  "name, and optionally a template." % bits[0])