def _loader(self, tags):
        """Create a yaml Loader."""
        class ConfigLoader(SafeLoader):
            pass
        ConfigLoader.add_multi_constructor("", lambda loader, prefix, node: TaggedValue(node.value, node.tag, *tags))
        return ConfigLoader