def rm(config, name, bucket):
    """ Delete lambda function, role, and zipfile """
    # options should override config if it is there
    myname = name or config.name
    mybucket = bucket or config.bucket

    click.echo('Deleting {} from {}'.format(myname, mybucket))
    lambder.delete_function(myname, mybucket)