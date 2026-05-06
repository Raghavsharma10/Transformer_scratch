def load(self, data, many=None, partial=None):
        """Deserialize a data structure to an object."""
        result = super(ResumptionTokenSchema, self).load(
            data, many=many, partial=partial
        )
        result.data.update(
            result.data.get('resumptionToken', {}).get('kwargs', {})
        )
        return result