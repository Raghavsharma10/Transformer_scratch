def offer_simple(pool, answer, rationale, student_id, options):
    """
    The simple selection algorithm.

    This algorithm randomly select an answer from the pool to discard and add the new one when the pool reaches
    the limit
    """
    existing = pool.setdefault(answer, {})
    if len(existing) >= get_max_size(pool, len(options), POOL_ITEM_LENGTH_SIMPLE):
        student_id_to_remove = random.choice(existing.keys())
        del existing[student_id_to_remove]
    existing[student_id] = {}
    pool[answer] = existing