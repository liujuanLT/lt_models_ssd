import numpy as np

class baseModel():
    def __init__(self):
        self.model = None

    def name(self):
        return self.__class__.__name__

    def preprocess(self,*args,**kwargs):
        print('args=',args)
        print('kwargs=',kwargs)
        raise NotImplementedError('please define preprocess method')

    def postprocess(self,*args,**kwargs):
        print('args=',args)
        print('kwargs=',kwargs)
        raise NotImplementedError('please define post process method')

    def load(self,*args,**kwargs):
        print('args=',args)
        print('kwargs=',kwargs)
        raise NotImplementedError('please define load method')

    def predict(self,data:np.ndarray,*args,**kwargs):
        print('args=',args)
        print('kwargs=',kwargs)
        raise NotImplementedError('please define predict method')

    def evaluate(self,*args,**kwargs):
        print('args=',args)
        print('kwargs=',kwargs)
        raise NotImplementedError('please define evaluate method')

class baseDataset():
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
   
    
def testDemo():
    model_path=''
    dataset=[]
    model = baseModel()
    model.load(model_path)
    for img,label in dataset:
        tmpimg = model.preprocess(img)
        result = model.predict(tmpimg)
        output = model.postprocess(result)
        model.evaluate(label, output)


    