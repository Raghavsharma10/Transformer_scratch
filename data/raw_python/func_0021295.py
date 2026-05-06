def parse_cell(cell, rules):
    """ Applies the rules to the bunch of text describing a cell.
    @param string cell
        A network / cell from iwlist scan.
    @param dictionary rules
        A dictionary of parse rules.

    @return dictionary
        parsed networks. """

    parsed_cell = {}
    for key in rules:
        rule = rules[key]
        parsed_cell.update({key: rule(cell)})
    return parsed_cell