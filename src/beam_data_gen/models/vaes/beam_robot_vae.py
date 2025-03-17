import torch

from vae_planner.models.vae_base import VaeBase, LatentVarsBase, TrainParams

from beam_data_gen.models.parameters.beam_vae_params import BeamVaeParams
from beam_data_gen.models.containers.beam_robot_containers import BeamRobotInputs, BeamRobotLatents, BeamRobotOutputs
from beam_data_gen.models.encoders.beam_robot_encoder import BeamRobotEncoder
from beam_data_gen.models.decoders.beam_decoder import BeamDecoder
from beam_data_gen.models.classifiers.beam_graph_classifier import BeamGraphClassifier


class BeamRobotVae(VaeBase):
    def __init__(self, 
                 vae_params: BeamVaeParams, 
                 train_params: TrainParams, 
                 encoder: BeamRobotEncoder, 
                 decoder: BeamDecoder,
                 classifier: BeamGraphClassifier):
        super().__init__(vae_params, train_params, encoder, decoder)
        
        self._vae_inputs = BeamRobotInputs()
        self._latents = BeamRobotLatents()
        
        self._classifier = classifier(self.vae_params).to(self.vae_params.device)
        
    def classifier(self, inputs: torch.tensor):
        return self._classifier.forward(inputs)
    
    def forward(self, inputs: BeamRobotInputs):
        super().forward(inputs)        
        self._vae_outputs.graph_edge_logits = self._classifier.forward(self._latents.z)
        return (self._latents, self._vae_outputs)
    
    def loss_func(self, inputs: BeamRobotInputs, latents: BeamRobotLatents, outputs: BeamRobotLatents):
        self._loss = super().loss_func(inputs, latents, outputs)       
        self._loss.cross_entropy = self._classifier.loss_func(self._vae_outputs.graph_edge_logits, inputs.graph_edge_targets)
        # Update the total loss with the weight
        self._loss.tot_loss = self._loss.tot_loss + self.train_params.bce * self._loss.cross_entropy
        return self._loss
