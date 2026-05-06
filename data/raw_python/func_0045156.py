def standalone(body):
    """ Returns complete html document given markdown html """
    with open(_ROOT + '/html.dat', 'r') as html_template:
        head = html_title()
        html = "".join(html_template.readlines()) \
                 .replace("{{HEAD}}", head) \
                 .replace("{{BODY}}", body)
        return html