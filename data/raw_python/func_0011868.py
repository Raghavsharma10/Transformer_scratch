def _parse_output_records(self, item: IterationRecord) -> Dict[str, Any]:
        """Parse output records into dicts ready for JSON."""
        output_records = {}
        for key, sub_item in item.output_records.items():
            if isinstance(sub_item, dict) or isinstance(sub_item, list):
                output_records[key] = sub_item
            else:
                output_records[key] = sub_item.__dict__

        return output_records