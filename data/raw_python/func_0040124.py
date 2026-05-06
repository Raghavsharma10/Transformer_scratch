def _nested_lookup(document, references, operation):
        """Lookup a key in a nested document, yield a value"""
        if isinstance(document, list):
            for d in document:
                for result in NestedLookup._nested_lookup(d, references, operation):
                    yield result

        if isinstance(document, dict):
            for k, v in document.items():
                if operation(k, v):
                    references.append((document, k))
                    yield v
                elif isinstance(v, dict):
                    for result in NestedLookup._nested_lookup(v, references, operation):
                        yield result
                elif isinstance(v, list):
                    for d in v:
                        for result in NestedLookup._nested_lookup(d, references, operation):
                            yield result