def placeholders(cls,dic):
        """Placeholders for fields names and value binds"""
        keys = [str(x) for x in dic]
        entete = ",".join(keys)
        placeholders = ",".join(cls.named_style.format(x) for x in keys)
        entete = f"({entete})"
        placeholders = f"({placeholders})"
        return entete, placeholders