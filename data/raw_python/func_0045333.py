def backend(self):
    "Internal property that returns the Node script running harness"
    if self._backend is None:
      with io.open(path.join(path.abspath(path.dirname(__file__)), "runner.js")) as runner_file:
        runner_source = runner_file.read()
      self._backend = execjs.ExternalRuntime(name="Node.js (V8)",
                                             command=["node"],
                                             runner_source=runner_source)
    return self._backend