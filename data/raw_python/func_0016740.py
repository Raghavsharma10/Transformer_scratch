def reformat_record(record):
    """Repack a record into a cleaner structure for consumption."""
    return {
        "key": record["dynamodb"].get("Keys", None),
        "new": record["dynamodb"].get("NewImage", None),
        "old": record["dynamodb"].get("OldImage", None),

        "meta": {
            "created_at": record["dynamodb"]["ApproximateCreationDateTime"],
            "event": {
                "id": record["eventID"],
                "type": record["eventName"].lower(),
                "version": record["eventVersion"]
            },
            "sequence_number": record["dynamodb"]["SequenceNumber"],
        }
    }