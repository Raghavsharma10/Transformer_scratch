def write_text(_command, txt_file):
    """Dump SQL command to a text file."""
    command = _command.strip()
    with open(txt_file, 'w') as txt:
        txt.writelines(command)