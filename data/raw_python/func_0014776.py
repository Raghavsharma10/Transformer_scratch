def _parse_string_to_nested_lists(self, value):
        """e.g., convert "[[a,b,c],[d,e],[f,g]]"
        into [["a","b","c"],["d","e"],["f","g"]]
        """
        if not re.match(r'\[.*\]', value.strip()):
            if '[' in value or ']' in value or ',' in value:
                raise Exception('Missing outer brace')
            elif len(value.strip()) == 0:
                raise Exception('Missing value')
            else:
                terms = value.split(',')
                terms = [term.strip() for term in terms]
                if len(terms) == 1:
                    return terms[0]
                else:
                    return terms

        # remove outer braces
        value = value[1:-1]
        terms = []
        depth = 0
        leftmost = 0
        first_open_brace = None
        break_on_commas = False
        for i in range(len(value)):
            if value[i] == ',' and depth == 0:
                terms.append(
                    self._parse_string_to_nested_lists(value[leftmost:i]))
                leftmost = i+1
            if value[i] == '[':
                if first_open_brace is None:
                    first_open_brace = i
                depth += 1
            if value[i] == ']':
                depth -= 1
                if depth < 0:
                    raise Exception('Unbalanced close brace')
            i += i
        if depth > 0:
            raise Exception('Expected "]"')
        terms.append(
            self._parse_string_to_nested_lists(value[leftmost:len(value)]))
        return terms