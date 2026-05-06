def vis_init(self):
        '''
        Sends the state of the BTC at the time the visualizer connects,
        initializing it.
        '''
        init_dict = {}
        init_dict['kind'] = 'init'
        assert len(self.want_file_pos) == len(self.heads_and_tails)
        init_dict['want_file_pos'] = self.want_file_pos
        init_dict['files'] = self.file_list
        init_dict['heads_and_tails'] = self.heads_and_tails
        init_dict['num_pieces'] = self.num_pieces
        self.broadcast(init_dict)