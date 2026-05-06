def create_helping_material_info(helping):
    """Create helping_material_info field."""
    helping_info = None
    file_path = None
    if helping.get('info'):
        helping_info = helping['info']
    else:
        helping_info = helping
    if helping_info.get('file_path'):
        file_path = helping_info.get('file_path')
        del helping_info['file_path']
    return helping_info, file_path