def _diffSchema(diskSchema, memorySchema):
    """
    Format a schema mismatch for human consumption.

    @param diskSchema: The on-disk schema.

    @param memorySchema: The in-memory schema.

    @rtype: L{bytes}
    @return: A description of the schema differences.
    """
    diskSchema = set(diskSchema)
    memorySchema = set(memorySchema)
    diskOnly = diskSchema - memorySchema
    memoryOnly = memorySchema - diskSchema
    diff = []
    if diskOnly:
        diff.append('Only on disk:')
        diff.extend(map(repr, diskOnly))
    if memoryOnly:
        diff.append('Only in memory:')
        diff.extend(map(repr, memoryOnly))
    return '\n'.join(diff)