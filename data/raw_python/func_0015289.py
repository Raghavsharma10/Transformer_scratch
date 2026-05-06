def _get_snippet_ctime(self, snip_name):
        """Returns and remembers (during this DevAssistant invocation) last ctime of given
        snippet.

        Calling ctime costs lost of time and some snippets, like common_args, are used widely,
        so we don't want to call ctime bazillion times on them during one invocation.

        Args:
            snip_name: name of snippet to get ctime for
        Returns:
            ctime of the snippet
        """
        if snip_name not in self.snip_ctimes:
            snippet = yaml_snippet_loader.YamlSnippetLoader.get_snippet_by_name(snip_name)
            self.snip_ctimes[snip_name] = os.path.getctime(snippet.path)
        return self.snip_ctimes[snip_name]