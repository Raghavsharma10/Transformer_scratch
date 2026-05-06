def assert_ast_like(sample, template, _path=None):
    """Check that the sample AST matches the template.
    
    Raises a suitable subclass of :exc:`ASTMismatch` if a difference is detected.
    
    The ``_path`` parameter is used for recursion; you shouldn't normally pass it.
    """
    if _path is None:
        _path = ['tree']

    if callable(template):
        # Checker function at the top level
        return template(sample, _path)

    if not isinstance(sample, type(template)):
        raise ASTNodeTypeMismatch(_path, sample, template)

    for name, template_field in ast.iter_fields(template):
        sample_field = getattr(sample, name)
        field_path = _path + [name]
        
        if isinstance(template_field, list):
            if template_field and (isinstance(template_field[0], ast.AST)
                                     or callable(template_field[0])):
                _check_node_list(field_path, sample_field, template_field)
            else:
                # List of plain values, e.g. 'global' statement names
                if sample_field != template_field:
                    raise ASTPlainListMismatch(field_path, sample_field, template_field)

        elif isinstance(template_field, ast.AST):
            assert_ast_like(sample_field, template_field, field_path)
        
        elif callable(template_field):
            # Checker function
            template_field(sample_field, field_path)

        else:
            # Single value, e.g. Name.id
            if sample_field != template_field:
                raise ASTPlainObjMismatch(field_path, sample_field, template_field)