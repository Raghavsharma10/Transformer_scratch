def json(self):
    """A JSON-encoded description of this board.

    Format:
    {'id': board_id,
     'title': 'The title of the board',
     'panels': [{
       'title': 'The title of the panel'
       'data_source': {
         'source_type': PanelSource.TYPE,
         'refresh_seconds': 600,
         ...source_specific_details...
       },
       'display': {
         'display_type': PanelDisplay.TYPE,
         ...display_specific_details...
       }, ...]}
    """
    if self.board_data:
      board_dict = json.loads(self.board_data)
      board_dict['id'] = self.id
      del board_dict['__version__']
    else:
      board_dict = {
        'id': self.id,
        'title': '',
        'panels': []
      }
    return board_dict
    """    pycode = self.pycodes.first() or PyCode()
    return {'id': self.id,
            'pycode': pycode.json()}
            """