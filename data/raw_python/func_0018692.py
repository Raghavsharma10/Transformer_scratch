def _load_blocks(self, blocks: List[Tuple[Key, Any]]) -> List[TimeAggregate]:
        """
        Converts [(Key, block)] to [BlockAggregate]
        :param blocks: List of (Key, block) blocks.
        :return: List of BlockAggregate
        """
        return [
            TypeLoader.load_item(self._schema.source.type)(self._schema.source, self._identity,
                                                           EvaluationContext()).run_restore(block)
            for (_, block) in blocks
        ]