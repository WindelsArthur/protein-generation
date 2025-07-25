import numpy as np

class NoiseSchedule:
    def __init__(self, schedule_type='cosine'):
        self.schedule_type = schedule_type
    
    def get_noise_level(self, t):
        if self.schedule_type == 'linear':
            return t
        elif self.schedule_type == 'cosine':
            return 1 - np.cos(t * np.pi / 2)
        elif self.schedule_type == 'sqrt':
            return np.sqrt(t)