import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
from core.datasets.CAREDrosophila import CAREDrosophilaDataset
from core.models.trunks.wnet import WNet
from core.models.add_uncertainty import add_uncertainty
from torch.utils.data import Dataset, DataLoader
import yaml
import pdb



if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + '/config.yml') as file:
      config = yaml.safe_load(file)
    pdb.set_trace()
    path = '/clusterfs/abc/angelopoulos/care/Isotropic_Drosophila/train_data/data_label.npz'
    dataset = CAREDrosophilaDataset(path, num_instances=20, normalize='min-max')
    model = WNet(1,1)
    model = add_uncertainty(model, config)
