def invoke(config, name, input):
    """ Invoke function in AWS """
    # options should override config if it is there
    myname = name or config.name

    click.echo('Invoking ' + myname)
    output = lambder.invoke_function(myname, input)
    click.echo(output)