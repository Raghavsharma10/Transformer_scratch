def millipede(size, comment=None, reverse=False, template='default', position=0, opposite=False):
    """
    Output the millipede
    """
    padding_offsets = [2, 1, 0, 1, 2, 3, 4, 4, 3]
    padding_suite_length = len(padding_offsets)
    head_padding_extra_offset = 2

    if opposite:
        padding_offsets.reverse()

    position = position or 0

    templates = {
        'frozen': {'bodyr': '╔═(❄❄❄)═╗', 'body': '╚═(❄❄❄)═╝',
                   'headr': '╔⊙ ⊙╗', 'head': '╚⊙ ⊙╝'},
        'love': {'bodyr': '╔═(♥♥♥)═╗', 'body': '╚═(♥♥♥)═╝',
                 'headr': '╔⊙ ⊙╗', 'head': '╚⊙ ⊙╝'},
        'corporate': {'bodyr': '╔═(©©©)═╗', 'body': '╚═(©©©)═╝',
                      'headr': '╔⊙ ⊙╗', 'head': '╚⊙ ⊙╝'},
        'musician': {'bodyr': '╔═(♫♩♬)═╗', 'body': '╚═(♫♩♬)═╝',
                     'headr': '╔⊙ ⊙╗', 'head': '╚⊙ ⊙╝'},
        'bocal': {'bodyr': '╔═(🐟🐟🐟)═╗', 'body': '╚═(🐟🐟🐟)═╝',
                  'headr': '╔⊙ ⊙╗', 'head': '╚⊙ ⊙╝'},
        'ascii': {'bodyr': '|=(###)=|', 'body': '|=(###)=|',
                  'headr': '/⊙ ⊙\\', 'head': '\\⊙ ⊙/'},
        'default': {'bodyr': '╔═(███)═╗', 'body': '╚═(███)═╝',
                    'headr': '╔⊙ ⊙╗', 'head': '╚⊙ ⊙╝'},
        'inception': {'bodyr': '╔═(🐛🐛🐛)═╗', 'body': '╚═(🐛🐛🐛)═╝',
                      'headr': '╔⊙ ⊙╗', 'head': '╚⊙ ⊙╝'},
        'humancentipede': {'bodyr': '╔═(😷😷😷)═╗', 'body': '╚═(😷😷😷)═╝',
                           'headr': '╔⊙ ⊙╗', 'head': '╚⊙ ⊙╝'},
        'heart': {'bodyr': '╔═(❤️❤️❤️)═╗', 'body': '╚═(❤️❤️❤️)═╝',
                  'headr': '╔⊙ ⊙╗', 'head': '╚⊙ ⊙╝'},
    }

    template = templates.get(template, templates['default'])

    head = "{}{}\n".format(
        " " * (padding_offsets[position % padding_suite_length] + head_padding_extra_offset),
        template['headr'] if reverse else template['head']
    )

    body_lines = [
        "{}{}\n".format(
            " " * padding_offsets[(x + position) % padding_suite_length],
            template['bodyr'] if reverse else template['body']
        )
        for x in range(size)
    ]

    if reverse:
        body_lines.reverse()

    body = "".join(body_lines)

    output = ""
    if reverse:
        output += body + head
        if comment:
            output += "\n" + comment + "\n"
    else:
        if comment:
            output += comment + "\n\n"
        output += head + body

    return output