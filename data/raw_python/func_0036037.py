def hitsquad(ctx):
    """'put a hit out' on all known rss feeds [Default action without arguements]"""
    with Database("feeds") as feeds:
        for name, feed in zip(list(feeds.keys()), list(feeds.values())):
            logger.debug("calling put_a_hit_out: %s", name)
            ctx.invoke(put_a_hit_out, name=name)
        if len(list(feeds.keys())) == 0:
            ctx.get_help()