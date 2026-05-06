def _build_opr_data(self, data, store):
        """Returns a well formatted OPR data"""
        return {
            "invoice_data": {
                "invoice": {
                    "total_amount": data.get("total_amount"),
                    "description": data.get("description")
                },
                "store": store.info
            },
            "opr_data": {
                "account_alias": data.get("account_alias")
            }
        }