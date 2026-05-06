def get_nearest_by_coordinates(
        data: list, latitude_key: str, longitude_key: str,
        target_latitude: float, target_longitude: float) -> Any:
    """Get the closest dict entry based on latitude/longitude."""
    return min(
        data,
        key=lambda p: haversine(
            target_latitude,
            target_longitude,
            float(p[latitude_key]),
            float(p[longitude_key])
        ))