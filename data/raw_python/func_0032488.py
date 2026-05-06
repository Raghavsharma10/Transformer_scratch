def dumpgrants(destination, as_json=None, setspec=None):
    """Harvest grants from OpenAIRE and store them locally."""
    if os.path.isfile(destination):
        click.confirm("Database '{0}' already exists."
                      "Do you want to write to it?".format(destination),
                      abort=True)  # no cover
    dumper = OAIREDumper(destination,
                         setspec=setspec)
    dumper.dump(as_json=as_json)