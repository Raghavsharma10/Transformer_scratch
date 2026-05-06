def compile(self, source, options={}):
    """Compile stylus into css

    source: A string containing the stylus code
    options: A dictionary of arguments to pass to the compiler

    Returns a string of css resulting from the compilation
    """
    options = dict(options)
    if "paths" in options:
      options["paths"] += self.paths
    else:
      options["paths"] = self.paths

    if "compress" not in options:
      options["compress"] = self.compress

    return self.context.call("compiler", source, options, self.plugins, self.imports)