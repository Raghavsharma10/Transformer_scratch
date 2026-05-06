def get_assistants_file_hierarchy(cls, dirs):
        """Returns assistants file hierarchy structure (see below) representing assistant
        hierarchy in given directories.

        It works like this:
        1. It goes through all *.yaml files in all given directories and adds them into
           hierarchy (if there are two files with same name in more directories, the file
           from first directory wins).
        2. For each {name}.yaml file, it calls itself recursively for {name} subdirectories
           of all given directories.

        Args:
            dirs: directories to search
        Returns:
            hierarchy structure that looks like this:
            {'assistant1':
                {'source': '/path/to/assistant1.yaml',
                 'subhierarchy': {<hierarchy of subassistants>}},
             'assistant2':
                {'source': '/path/to/assistant2.yaml',
                 'subhierarchy': {<another hierarchy of subassistants}}
            }
        """
        result = {}
        for d in filter(lambda d: os.path.exists(d), dirs):
            for f in filter(lambda f: f.endswith('.yaml'), os.listdir(d)):
                assistant_name = f[:-5]
                if assistant_name not in result:
                    subas_dirs = [os.path.join(dr, assistant_name) for dr in dirs]
                    result[assistant_name] = {'source': os.path.join(d, f),
                                              'subhierarchy':
                                              cls.get_assistants_file_hierarchy(subas_dirs)}

        return result