class BaseAgent():
    def __init__(self, config):
        self.config = config

    def reset(self, **kwargs):
        raise NotImplementedError
    
    def observe_without_action(self, obs):
        raise NotImplementedError

    def response_answer(self, question, choices, time):
        raise NotImplementedError

    def response_retri(self, question, choices, time):
        raise NotImplementedError
    
    def process(self):
        raise NotImplementedError
