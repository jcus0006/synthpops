import json
import numpy as np
from synthpops import tourism as trsm

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, trsm.Tourist):
            return obj.__dict__
        if isinstance(obj, trsm.TouristGroup):
            return obj.__dict__
        return super().default(obj)