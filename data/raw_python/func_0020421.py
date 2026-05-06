def switch_toggle(context, ain):
    """Toggle an actor's power state"""
    context.obj.login()
    actor = context.obj.get_actor_by_ain(ain)
    if actor:
        if actor.get_state():
            actor.switch_off()
            click.echo("State for {} is now OFF".format(ain))
        else:
            actor.switch_on()
            click.echo("State for {} is now ON".format(ain))
    else:
        click.echo("Actor not found: {}".format(ain))