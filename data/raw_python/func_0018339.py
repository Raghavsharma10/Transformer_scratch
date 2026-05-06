def print_action_groups(data, nested_content, markDownHelp=False, settings=None):
    """
    Process all 'action groups', which are also include 'Options' and 'Required
    arguments'. A list of nodes is returned.
    """
    definitions = map_nested_definitions(nested_content)
    nodes_list = []
    if 'action_groups' in data:
        for action_group in data['action_groups']:
            # Every action group is comprised of a section, holding a title, the description, and the option group (members)
            section = nodes.section(ids=[action_group['title']])
            section += nodes.title(action_group['title'], action_group['title'])

            desc = []
            if action_group['description']:
                desc.append(action_group['description'])
            # Replace/append/prepend content to the description according to nested content
            subContent = []
            if action_group['title'] in definitions:
                classifier, s, subContent = definitions[action_group['title']]
                if classifier == '@replace':
                    desc = [s]
                elif classifier == '@after':
                    desc.append(s)
                elif classifier == '@before':
                    desc.insert(0, s)
                elif classifier == '@skip':
                    continue
                if len(subContent) > 0:
                    for k, v in map_nested_definitions(subContent).items():
                        definitions[k] = v
            # Render appropriately
            for element in renderList(desc, markDownHelp):
                section += element

            localDefinitions = definitions
            if len(subContent) > 0:
                localDefinitions = {k: v for k, v in definitions.items()}
                for k, v in map_nested_definitions(subContent).items():
                    localDefinitions[k] = v

            items = []
            # Iterate over action group members
            for entry in action_group['options']:
                """
                Members will include:
                    default	The default value. This may be ==SUPPRESS==
                    name	A list of option names (e.g., ['-h', '--help']
                    help	The help message string
                There may also be a 'choices' member.
                """
                # Build the help text
                arg = []
                if 'choices' in entry:
                    arg.append('Possible choices: {}\n'.format(", ".join([str(c) for c in entry['choices']])))
                if 'help' in entry:
                    arg.append(entry['help'])
                if entry['default'] is not None and entry['default'] not in ['"==SUPPRESS=="', '==SUPPRESS==']:
                    if entry['default'] == '':
                        arg.append('Default: ""')
                    else:
                        arg.append('Default: {}'.format(entry['default']))

                # Handle nested content, the term used in the dict has the comma removed for simplicity
                desc = arg
                term = ' '.join(entry['name'])
                if term in localDefinitions:
                    classifier, s, subContent = localDefinitions[term]
                    if classifier == '@replace':
                        desc = [s]
                    elif classifier == '@after':
                        desc.append(s)
                    elif classifier == '@before':
                        desc.insert(0, s)
                term = ', '.join(entry['name'])

                n = nodes.option_list_item('',
                                           nodes.option_group('', nodes.option_string(text=term)),
                                           nodes.description('', *renderList(desc, markDownHelp, settings)))
                items.append(n)

            section += nodes.option_list('', *items)
            nodes_list.append(section)

    return nodes_list