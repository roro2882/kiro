class Env():
    hello = "coucou"
    def __init__(self, config):
        return;

    def reset(self):
        state = self.hello
        return state

    def step(self, action):
        return 0;
        #return  self.etat, reward, done, ""
