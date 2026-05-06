def tag(context: click.Context, file_id: int, tags: List[str]):
    """Add tags to an existing file."""
    file_obj = context.obj['db'].file_(file_id)
    if file_obj is None:
        print(click.style('unable to find file', fg='red'))
        context.abort()
    for tag_name in tags:
        tag_obj = context.obj['db'].tag(tag_name)
        if tag_obj is None:
            tag_obj = context.obj['db'].new_tag(tag_name)
        elif tag_obj in file_obj.tags:
            print(click.style(f"{tag_name}: tag already added", fg='yellow'))
            continue
        file_obj.tags.append(tag_obj)
    context.obj['db'].commit()
    all_tags = (tag.name for tag in file_obj.tags)
    print(click.style(f"file tags: {', '.join(all_tags)}", fg='blue'))