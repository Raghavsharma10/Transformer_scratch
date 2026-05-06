def switch_on(context, ain):
    """Switch an actor's power to ON"""
    context.obj.login()
    actor = context.obj.get_actor_by_ain(ain)
    if actor:
        click.echo("Switching {} on".format(actor.name))
        actor.switch_on()
    else:
        click.echo("Actor not found: {}".format(ain))