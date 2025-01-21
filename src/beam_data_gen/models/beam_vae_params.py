import torch

from vae_planner.parameter_handlers.vae_params import VaeParams


class BeamVaeParams(VaeParams):
    def __init__(self, args=None):
        super().__init__(args)
        
        # Beam quantities
        self.no_beams = 3 # Number of beams 
        self.beam_latent_dim = 10 # Latent dim size of each latent dimension
        
        # Robot quantities
        self.no_hands = 2
        self.robot_latent_dim = 10
        
        self.action_dim = 0 # Action size

        self.state_dim = 5 # State space dimension of the robot

        self.no_inputs = 1
        self.no_outputs = 1
        
        self.pos_lims = [0.60, 0.60, 0.08]
        
        self.no_classifier_nodes = 3 # self.no_classifier_pred = self.no_classifier_nodes**2

        self.in_path = ""
        self.out_dir = ""

        self.data_freq = 50.0 # Hz sampling frequency of the data

        self.enc_freq = 50.0 # Hz encoder frequency
        self.dec_freq = 50.0 # Hz decoder frequency

        self.no_recurrent_steps = 0 # Number of recurrent repeats

        self.model_width = 256 # Width of the model

        self.batch_first = False # Torch puts the batch after the trajectory length

        self.cuda = True # Use cuda or cpu

        if args is not None:
            self.set_from_args(args)


    def set_from_args(self, args) -> None:
        
        self.beam_latent_dim = args.beam_latent_dim
        self.no_beams = args.no_beams
        
        self.no_hands = args.no_hands
        self.robot_latent_dim = args.robot_latent_dim
        
        self.pos_lims = args.pos_lims
        
        self.action_dim = args.action_dim
        self.no_classifier_nodes = args.no_classifier_nodes
        self.state_dim = args.state_dim
        self.in_path = args.in_path
        self.out_dir = args.out_dir
        self.no_inputs = args.no_inputs
        self.no_outputs = args.no_outputs
        self.data_freq = args.data_freq
        self.enc_freq = args.enc_freq
        self.dec_freq = args.dec_freq
        self.no_recurrent_steps = args.no_recurrent_steps
        self.model_width = args.model_width
        self.batch_first = args.batch_first
        self.cuda = args.cuda
        
    @property
    def latent_dim(self) -> int:
        self._latent_dim = self.beam_latent_dim + self.robot_latent_dim
        return self._latent_dim

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if self.cuda else "cpu")
    
    @property
    def input_dim(self) -> int:
        return (self.state_dim * self.no_inputs * self.no_beams) + \
                (self.state_dim * self.no_inputs * self.no_hands)
    
    @property
    def output_dim(self) -> int:
        return (self.state_dim * self.no_outputs * self.no_beams) + \
                (self.state_dim * self.no_outputs * self.no_hands)
                
    @property
    def beam_input_dim(self) -> int:
        return self.state_dim * self.no_beams * self.no_inputs
    
    @property
    def robot_input_dim(self) -> int:
        return self.state_dim * self.no_hands * self.no_inputs
    
    @property
    def decoder_input_dim(self) -> int:
        return self.latent_dim + self.action_dim

    @property
    def no_samps_to_skip(self) -> int: 
        return int(self.dec_freq / self.enc_freq)
    
    @property
    def no_classifier_pred(self):
        return int(self.no_classifier_nodes ** 2.0)
        
