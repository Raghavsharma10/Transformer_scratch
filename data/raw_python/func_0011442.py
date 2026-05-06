def finish_user(self, subid):
    '''finish user will remove a user's token, making the user entry not
       accesible if running in headless model'''        

    p = self.revoke_token(subid)
    p.token = "finished"
    self.session.commit()
    return p