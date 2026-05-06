def _apply(self, key_value_map):
    """Apply the filter to values extracted from an entity.

    Think of self.match_keys and self.match_values as representing a
    table with one row.  For example:

      match_keys = ('name', 'age', 'rank')
      match_values = ('Joe', 24, 5)

    (Except that in reality, the values are represented by tuples
    produced by datastore_types.PropertyValueToKeyValue().)

    represents this table:

      |  name   |  age  |  rank  |
      +---------+-------+--------+
      |  'Joe'  |   24  |     5  |

    Think of key_value_map as a table with the same structure but
    (potentially) many rows.  This represents a repeated structured
    property of a single entity.  For example:

      {'name': ['Joe', 'Jane', 'Dick'],
       'age': [24, 21, 23],
       'rank': [5, 1, 2]}

    represents this table:

      |  name   |  age  |  rank  |
      +---------+-------+--------+
      |  'Joe'  |   24  |     5  |
      |  'Jane' |   21  |     1  |
      |  'Dick' |   23  |     2  |

    We must determine wheter at least one row of the second table
    exactly matches the first table.  We need this class because the
    datastore, when asked to find an entity with name 'Joe', age 24
    and rank 5, will include entities that have 'Joe' somewhere in the
    name column, 24 somewhere in the age column, and 5 somewhere in
    the rank column, but not all aligned on a single row.  Such an
    entity should not be considered a match.
    """
    columns = []
    for key in self.match_keys:
      column = key_value_map.get(key)
      if not column:  # None, or an empty list.
        return False  # If any column is empty there can be no match.
      columns.append(column)
    # Use izip to transpose the columns into rows.
    return self.match_values in itertools.izip(*columns)