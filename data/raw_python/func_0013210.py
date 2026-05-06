def marshall(self, registry):
        """Marshalls a full registry (various collectors)"""

        blocks = []
        for i in registry.get_all():
            blocks.append(self.marshall_collector(i))

        # Sort? used in tests
        blocks = sorted(blocks)

        # Needs EOF
        blocks.append("")

        return self.__class__.LINE_SEPARATOR_FMT.join(blocks)