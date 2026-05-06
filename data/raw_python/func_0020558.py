def get_dock_json(self):
        """ return dock json from existing build json """
        env_json = self.build_json['spec']['strategy']['customStrategy']['env']
        try:
            p = [env for env in env_json if env["name"] == "ATOMIC_REACTOR_PLUGINS"]
        except TypeError:
            raise RuntimeError("\"env\" is not iterable")
        if len(p) <= 0:
            raise RuntimeError("\"env\" misses key ATOMIC_REACTOR_PLUGINS")
        dock_json_str = p[0]['value']
        dock_json = json.loads(dock_json_str)
        return dock_json