def config(ctx):
    """[GROUP] Configuration management operations"""

    from hfos import database
    database.initialize(ctx.obj['dbhost'], ctx.obj['dbname'])

    from hfos.schemata.component import ComponentConfigSchemaTemplate
    ctx.obj['col'] = model_factory(ComponentConfigSchemaTemplate)