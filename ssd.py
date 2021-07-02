import numpy as np
import onnxruntime
import torch
from torch.utils.data import DataLoader
import baseModel
from cocodataset import CocoDataset


def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class SSD(baseModel.baseModel):
    def __init__(self, cfg_path):
        super(SSD, self).__init__()
        self.model = None
        self._backend = None
        self.ort_session = None
    
    def name(self): # TODO
        return self.__class__.__name__

    def preprocess(self):
        pass

    def postprocess(self,*args,**kwargs):
        pass

    def load(self, modelpath):    
        pass
        # TODO
        # self.model = None
        # self._backend = None
        # self.ort_session = onnxruntime.InferenceSession(modelpath)

    def predict(self, data:np.ndarray):
        if self._backend == "pytorch":
            outputs = self.model(data)
        elif self._backend == "onnxruntime":
            inputs = {self.ort_session.get_inputs()[0].name: data}
            outputs = self.ort_session.run(None, inputs)
            outputs = outputs[0] # TODO
    

    def evaluate(self, outputs, metrics='mAP', evals=['bbox']):
        pass
        # TODO
        # outputs: results for one sample
            


def test_ssd():
    cfg_path = "./ssd_cfg.py"
    model = SSD()
    model.load(cfg_path)
    dataset = CocoDataset.CocoDataset(cfg_path)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            tmpimg = model.preprocess(data['img'])
            result = model.predict(tmpimg)
            output = model.postprocess(result)
            model.evaluate(data['label'], output) # TODO
    



