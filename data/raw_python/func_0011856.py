def _repair(record: Dict[str, Any]) -> Dict[str, Any]:
    """Repair a corrupted IterationRecord with a specific known issue."""
    output_records = record.get("output_records")
    if record.get("_type", None) == "IterationRecord" and output_records is not None:
        birdsite_record = output_records.get("birdsite")

        # check for the bug
        if isinstance(birdsite_record, dict) and birdsite_record.get("_type") == "IterationRecord":

            # get to the bottom of the corrupted record
            failed = False
            while birdsite_record.get("_type") == "IterationRecord":
                sub_record = birdsite_record.get("output_records")
                if sub_record is None:
                    failed = True
                    break

                birdsite_record = sub_record.get("birdsite")
                if birdsite_record is None:
                    failed = True
                    break

            if failed:
                return record

            # add type
            birdsite_record["_type"] = TweetRecord.__name__

            # lift extra keys, just in case
            if "extra_keys" in birdsite_record:
                record_extra_values = record.get("extra_keys", {})
                for key, value in birdsite_record["extra_keys"].items():
                    if key not in record_extra_values:
                        record_extra_values[key] = value

                record["extra_keys"] = record_extra_values

                del birdsite_record["extra_keys"]

            output_records["birdsite"] = birdsite_record

        # pull that correct record up to the top level, fixing corruption
        record["output_records"] = output_records

    return record