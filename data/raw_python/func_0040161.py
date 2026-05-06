def validate(text, file, schema_type):
    """Validate JSON input using dependencies-schema"""
    content = None

    if text:
        print('Validating text input...')
        content = text

    if file:
        print('Validating file input...')
        content = file.read()

    if content is None:
        click.secho('Please give either text input or a file path. See help for more details.', fg='red')
        exit(1)

    try:
        if schema_type == 'dependencies':
            validator = DependenciesSchemaValidator()
        elif schema_type == 'actions':
            validator = ActionsSchemaValidator()
        else:
            raise Exception('Unknown type')

        validator.validate_json(content)
        click.secho('Valid JSON schema!', fg='green')
    except Exception as e:
        click.secho('Invalid JSON schema!', fg='red')
        raise e