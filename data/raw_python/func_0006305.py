def neighbors(geohash):
  """Return all neighboring geohashes."""
  return {
    'n':  adjacent(geohash, 'n'),
    'ne': adjacent(adjacent(geohash, 'n'), 'e'),
    'e':  adjacent(geohash, 'e'),
    'se': adjacent(adjacent(geohash, 's'), 'e'),
    's':  adjacent(geohash, 's'),
    'sw': adjacent(adjacent(geohash, 's'), 'w'),
    'w':  adjacent(geohash, 'w'),
    'nw': adjacent(adjacent(geohash, 'n'), 'w'),
    'c':  geohash
  }