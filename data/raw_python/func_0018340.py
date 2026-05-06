def print_subcommands(data, nested_content, markDownHelp=False, settings=None):
    """
    Each subcommand is a dictionary with the following keys:

    ['usage', 'action_groups', 'bare_usage', 'name', 'help']

    In essence, this is all tossed in a new section with the title 'name'.
    Apparently there can also be a 'description' entry.
    """

    definitions = map_nested_definitions(nested_content)
    items = []
    if 'children' in data:
        subCommands = nodes.section(ids=["Sub-commands:"])
        subCommands += nodes.title('Sub-commands:', 'Sub-commands:')

        for child in data['children']:
            sec = nodes.section(ids=[child['name']])
            sec += nodes.title(child['name'], child['name'])

            if 'description' in child and child['description']:
                desc = [child['description']]
            elif child['help']:
                desc = [child['help']]
            else:
                desc = ['Undocumented']

            # Handle nested content
            subContent = []
            if child['name'] in definitions:
                classifier, s, subContent = definitions[child['name']]
                if classifier == '@replace':
                    desc = [s]
                elif classifier == '@after':
                    desc.append(s)
                elif classifier == '@before':
                    desc.insert(0, s)

            for element in renderList(desc, markDownHelp):
                sec += element
            sec += nodes.literal_block(text=child['bare_usage'])
            for x in print_action_groups(child, nested_content + subContent, markDownHelp,
                                         settings=settings):
                sec += x

            for x in print_subcommands(child, nested_content + subContent, markDownHelp,
                                       settings=settings):
                sec += x

            if 'epilog' in child and child['epilog']:
                for element in renderList([child['epilog']], markDownHelp):
                    sec += element

            subCommands += sec
        items.append(subCommands)

    return items