def file_cmd(context, yes, file_id):
    """Delete a file."""
    file_obj = context.obj['store'].File.get(file_id)
    if file_obj.is_included:
        question = f"remove file from file system and database: {file_obj.full_path}"
    else:
        question = f"remove file from database: {file_obj.full_path}"
    if yes or click.confirm(question):
        if file_obj.is_included and Path(file_obj.full_path).exists():
            Path(file_obj.full_path).unlink()

        file_obj.delete()
        context.obj['store'].commit()
        click.echo('file deleted')