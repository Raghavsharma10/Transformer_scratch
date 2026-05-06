def context(self):
    "Internal property that returns the stylus compiler"
    if self._context is None:
      with io.open(path.join(path.abspath(path.dirname(__file__)), "compiler.js")) as compiler_file:
        compiler_source = compiler_file.read()
      self._context = self.backend.compile(compiler_source)
    return self._context