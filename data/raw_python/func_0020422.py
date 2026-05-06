def logs(context, format):
    """Show system logs since last reboot"""
    fritz = context.obj
    fritz.login()

    messages = fritz.get_logs()
    if format == "plain":
        for msg in messages:
            merged = "{} {} {}".format(msg.date, msg.time, msg.message.encode("UTF-8"))
            click.echo(merged)

    if format == "json":
        entries = [msg._asdict() for msg in messages]
        click.echo(json.dumps({
            "entries": entries,
        }))