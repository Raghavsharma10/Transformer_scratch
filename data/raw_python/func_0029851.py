def execute(self):
        """Convert the notebook to a python script and execute it, returning the local context
        as a dict"""

        from nbformat import read
        from nbconvert.exporters import export_script
        from cStringIO import StringIO

        notebook = read(StringIO(self.record.unpacked_contents), 4)

        script, resources = export_script(notebook)

        env_dict = {}

        exec (compile(script.replace('# coding: utf-8', ''), 'script', 'exec'), env_dict)

        return env_dict