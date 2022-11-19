import pickle

class StoreModels():
    def __init__(self, models, filename):
        self.models = models
        self.filename = filename

    def get_models(self):
        return self.models
    
    def get_filename(self):
        return self.filename
        
    def store(self, models):
        models = self.get_models()
        with open(self.get_filename, 'wb') as file:
            pickle.dump(models, file, protocol=pickle.HIGHEST_PROTOCOL)
            print("Models stored!")
        
        comparison = self.compare_models(self.get_filename(), self.get_models())
        print(comparison)        

    def compare_models(filename, model):
        equal = False
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            
            if data == model:
                equal = True
            else:
                print("Models are not equal!")
        return equal         
    