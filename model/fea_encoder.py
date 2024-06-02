from model.feature_octree import FeatureOctree
from model.feature_plane import FeaturePlane
from utils.config import Config

class FeatureEncoder:
    def __init__(self, config: Config):
        super().__init__()
        self.fea_encoder_type = config.fea_encoder_type
        self.fea_encoder = None
        if self.fea_encoder_type == 'fea_octree':
            self.fea_encoder = FeatureOctree(config)
        elif self.fea_encoder_type == 'fea_plane':
            self.fea_encoder = FeaturePlane(config)
        self.out_dim = self.fea_encoder.out_dim

    def forward(self, x):
        x = self.fea_encoder(x)
        return x
