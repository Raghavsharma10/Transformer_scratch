def energy(context, features):
    """Display energy stats of all actors"""
    fritz = context.obj
    fritz.login()

    for actor in fritz.get_actors():
        if actor.temperature is not None:
            click.echo("{} ({}): {:.2f} Watt current, {:.3f} wH total, {:.2f} °C".format(
                actor.name.encode('utf-8'),
                actor.actor_id,
                (actor.get_power() or 0.0) / 1000,
                (actor.get_energy() or 0.0) / 100,
                actor.temperature
            ))
        else:
            click.echo("{} ({}): {:.2f} Watt current, {:.3f} wH total, offline".format(
                actor.name.encode('utf-8'),
                actor.actor_id,
                (actor.get_power() or 0.0) / 1000,
                (actor.get_energy() or 0.0) / 100
            ))
        if features:
            click.echo("  Features: PowerMeter: {}, Temperatur: {}, Switch: {}".format(
                actor.has_powermeter, actor.has_temperature, actor.has_switch
            ))