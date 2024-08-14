import GraphBuilder
from GraphState import GraphState 
import json
import numpy as np 

class NetGAPStats:
    def __init__(self,evolved_state,best_state,stats):
        self.evolved_state=evolved_state
        self.best_state = best_state
        self.stats = stats
        
        for key in ['evolved_state','best_state']:   
            state = self.__dict__[key]
            if isinstance(state,GraphState):
                setattr(self, f'{key}_score' ,state.reward_stats['score'])
            elif isinstance(state,dict):
                setattr(self, f'{key}_score' ,state['reward_stats']['score'])
        
        for key in stats:
            setattr(self, key, stats[key])
        
    def __str__(self):
        return f"best: {self.best_state_score}, evolved: {self.evolved_state_score}, {self.stats}"
    
    def serialize(self):
        ret = {'evolved_state': self.evolved_state.serialize(),'best_state': self.best_state.serialize()}
        ret.update(self.stats)
        return ret
    
    @staticmethod
    def unserialize(serialized_object):
        evolved_state=serialized_object['evolved_state'] 
        best_state = serialized_object['best_state']
        stats = {k:s for k,s in serialized_object.items() if k not in ['evolved_state','best_state']}
        
        temp = NetGAPStats(evolved_state,best_state,stats)
        
        return temp

class ComparisonSaver:
    
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)
    
    def __init__(self,directory):
        self.directory=directory
        
    def save(self,history, filename):
        
        filename = f"{self.directory}\{filename}.json"

        data_list=[]
        for dp in history:
            x=dp.serialize()
            data_list.append(x)
           
        print("will write:", len(data_list), " data points to file ", filename)
        json_str=json.dumps(data_list, ensure_ascii = True, cls=self.NpEncoder)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_str, f, ensure_ascii=False, indent=4)
            f.close()
              
        return filename
    
    def load(self,filename):
        filename = f"{self.directory}\{filename}.json"
        serialized_list=[]
        with open(filename, 'r', encoding='utf-8') as fo:
            serialized_list = json.loads(json.load(fo))
            
        unserialized_list=[NetGAPStats.unserialize(s) for s in serialized_list]

        
        print("loaded:", len( unserialized_list), " data points from file ", filename)
        return  unserialized_list

