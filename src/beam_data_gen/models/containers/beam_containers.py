from vae_planner.models.container_base import (VaeInputsBase, LatentVarsBase, VaeOutputsBase, LossOutputsBase)


class BeamVaeInputs(VaeInputsBase):
    # Model inputs and targets
    def __init__(self):
        super().__init__()
        self.graph_edge_targets = None
    

class BeamVaeOutputs(VaeOutputsBase):
    # Model predictions
    def __init__(self):
        super().__init__()
        self.graph_edge_logits = None
    
