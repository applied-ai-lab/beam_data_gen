class TrainParams:
    def __init__(self, args=None) -> None:
        self.read_from_file = False
        self.output_to_file = True
        self.data_path = ""
        # optim_params:
        self.batch_size   = 256
        self.epochs       = 300
        self.seed         = 1
        self.log_interval = 10
        self.lr           = 1.0e-3
        # loss params:
        self.kl = 0.2
        self.bce = 0.1
        self.gce = 0.5
        self.vce = 0.0
        self.reduction = 'sum'
        self.bce_reduction = 'sum'
        self.ce_reduction = 'sum'       

        if args is not None:
            self.set_from_args(args)


    def set_from_args(self, args) -> None:
        self.read_from_file = args.read_from_file
        self.output_to_file = args.output_to_file
        self.data_path      = args.data_path
        self.batch_size     = args.batch_size
        self.epochs         = args.epochs
        self.seed           = args.seed
        self.log_interval   = args.log_interval
        self.lr             = args.lr
        self.kl             = args.kl
        self.bce            = args.bce
        self.gce            = args.gce
        self.vce            = args.vce
        self.reduction      = args.reduction
        self.bce_reduction  = args.bce_reduction
        self.ce_reduction   = args.ce_reduction

