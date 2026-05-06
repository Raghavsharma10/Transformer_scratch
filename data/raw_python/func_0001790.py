def serialize(obj):
    """JSON serializer that accepts datetime & date"""
    from datetime import datetime, date, time
    if isinstance(obj, date) and not isinstance(obj, datetime):
        obj = datetime.combine(obj, time.min)
    if isinstance(obj, datetime):
        return obj.isoformat()