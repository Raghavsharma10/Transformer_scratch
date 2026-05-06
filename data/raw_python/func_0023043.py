def _get_vispy_font_filename(face, bold, italic):
    """Fetch a remote vispy font"""
    name = face + '-'
    name += 'Regular' if not bold and not italic else ''
    name += 'Bold' if bold else ''
    name += 'Italic' if italic else ''
    name += '.ttf'
    return load_data_file('fonts/%s' % name)