def construct_arg(cls, name, params):
        """Construct an argument from name, and params (dict loaded from assistant/snippet).
        """
        use_snippet = params.pop('use', None)
        if use_snippet:
            # if snippet is used, take this parameter from snippet and update
            # it with current params, if any
            try:
                problem = None
                snippet = yaml_snippet_loader.YamlSnippetLoader.get_snippet_by_name(use_snippet)
                # this works much like snippet.args.pop(arg_name).update(arg_params),
                # but unlike it, this actually returns the updated dict
                params = dict(snippet.args.pop(name), **params)
                # if there is SnippetNotFoundException, just let it be raised
            except KeyError:  # snippet doesn't have the requested argument
                problem = 'Couldn\'t find arg {arg} in snippet {snip}.'.\
                    format(arg=name, snip=snippet.name)
                raise exceptions.ExecutionException(problem)

        if 'flags' not in params:
            msg = 'Couldn\'t find "flags" in arg {arg}'.format(arg=name)
            raise exceptions.ExecutionException(msg)
        return cls(name, *params.pop('flags'), **params)