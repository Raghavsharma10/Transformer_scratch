def renderList(l, markDownHelp, settings=None):
    """
    Given a list of reStructuredText or MarkDown sections, return a docutils node list
    """
    if len(l) == 0:
        return []
    if markDownHelp:
        from sphinxarg.markdown import parseMarkDownBlock
        return parseMarkDownBlock('\n\n'.join(l) + '\n')
    else:
        all_children = []
        for element in l:
            if isinstance(element, str):
                if settings is None:
                    settings = OptionParser(components=(Parser,)).get_default_values()
                document = new_document(None, settings)
                Parser().parse(element + '\n', document)
                all_children += document.children
            elif isinstance(element, nodes.definition):
                all_children += element

        return all_children