def detect_output_type(args):
    """Detect whether to save to a single or multiple files."""
    if not args['single'] and not args['multiple']:
        # Save to multiple files if multiple files/URLs entered
        if len(args['query']) > 1 or len(args['out']) > 1:
            args['multiple'] = True
        else:
            args['single'] = True