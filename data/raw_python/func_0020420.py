def switch_state(context, ain):
    """Get an actor's power state"""
    context.obj.login()
    actor = context.obj.get_actor_by_ain(ain)
    if actor:
        click.echo("State for {} is: {}".format(ain,'ON' if actor.get_state() else 'OFF'))
    else:
        click.echo("Actor not found: {}".format(ain))