def validate_table(self, table_name, model):
        """Polls until a creating table is ready, then verifies the description against the model's requirements.

        The model may have a subset of all GSIs and LSIs on the table, but the key structure must be exactly
        the same.  The table must have a stream if the model expects one, but not the other way around.  When read or
        write units are not specified for the model or any GSI, the existing values will always pass validation.

        :param str table_name: The name of the table to validate the model against.
        :param model: The :class:`~bloop.models.BaseModel` to validate the table of.
        :raises bloop.exceptions.TableMismatch: When the table does not meet the constraints of the model.
        """
        actual = self.describe_table(table_name)
        if not compare_tables(model, actual):
            raise TableMismatch("The expected and actual tables for {!r} do not match.".format(model.__name__))

        # Fill in values that Meta doesn't know ahead of time (such as arns).
        # These won't be populated unless Meta explicitly cares about the value
        if model.Meta.stream:
            stream_arn = model.Meta.stream["arn"] = actual["LatestStreamArn"]
            logger.debug(f"Set {model.__name__}.Meta.stream['arn'] to '{stream_arn}' from DescribeTable response")
        if model.Meta.ttl:
            ttl_enabled = actual["TimeToLiveDescription"]["TimeToLiveStatus"].lower() == "enabled"
            model.Meta.ttl["enabled"] = ttl_enabled
            logger.debug(f"Set {model.__name__}.Meta.ttl['enabled'] to '{ttl_enabled}' from DescribeTable response")

        # Fill in meta values that the table didn't care about (eg. billing=None)
        if model.Meta.encryption is None:
            sse_enabled = actual["SSEDescription"]["Status"].lower() == "enabled"
            model.Meta.encryption = {"enabled": sse_enabled}
            logger.debug(
                f"Set {model.__name__}.Meta.encryption['enabled'] to '{sse_enabled}' from DescribeTable response")
        if model.Meta.backups is None:
            backups = actual["ContinuousBackupsDescription"]["ContinuousBackupsStatus"] == "ENABLED"
            model.Meta.backups = {"enabled": backups}
            logger.debug(f"Set {model.__name__}.Meta.backups['enabled'] to '{backups}' from DescribeTable response")
        if model.Meta.billing is None:
            billing_mode = {
                "PAY_PER_REQUEST": "on_demand",
                "PROVISIONED": "provisioned"
            }[actual["BillingModeSummary"]["BillingMode"]]
            model.Meta.billing = {"mode": billing_mode}
            logger.debug(f"Set {model.__name__}.Meta.billing['mode'] to '{billing_mode}' from DescribeTable response")
        if model.Meta.read_units is None:
            read_units = model.Meta.read_units = actual["ProvisionedThroughput"]["ReadCapacityUnits"]
            logger.debug(
                f"Set {model.__name__}.Meta.read_units to {read_units} from DescribeTable response")
        if model.Meta.write_units is None:
            write_units = model.Meta.write_units = actual["ProvisionedThroughput"]["WriteCapacityUnits"]
            logger.debug(
                f"Set {model.__name__}.Meta.write_units to {write_units} from DescribeTable response")

        # Replace any ``None`` values for read_units, write_units in GSIs with their actual values
        gsis = {index["IndexName"]: index for index in actual["GlobalSecondaryIndexes"]}
        for index in model.Meta.gsis:
            read_units = gsis[index.dynamo_name]["ProvisionedThroughput"]["ReadCapacityUnits"]
            write_units = gsis[index.dynamo_name]["ProvisionedThroughput"]["WriteCapacityUnits"]
            if index.read_units is None:
                index.read_units = read_units
                logger.debug(
                    f"Set {model.__name__}.{index.name}.read_units to {read_units} from DescribeTable response")
            if index.write_units is None:
                index.write_units = write_units
                logger.debug(
                    f"Set {model.__name__}.{index.name}.write_units to {write_units} from DescribeTable response")