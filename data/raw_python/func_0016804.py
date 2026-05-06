def by_alias(cls, name: str) -> "TxType":
        """get a type by the common bloop operation name: get/check/delete/save"""
        return {
            "get": TxType.Get,
            "check": TxType.Check,
            "delete": TxType.Delete,
            "save": TxType.Update,
        }[name]