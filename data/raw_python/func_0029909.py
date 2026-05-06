def name_parts(self):
        """Works with PartialNameMixin.clear_dict to set NONE and ANY
        values."""

        default = PartialMixin.ANY

        return ([(k, default, True)
                for k, _, _ in PartitionName._name_parts]
                +
                [(k, default, True)
                 for k, _, _ in Name._generated_names]
                )