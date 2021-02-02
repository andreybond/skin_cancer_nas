sys.path.append('/mnt')
sys.path.append('/mnt/skin_cancer_nas')
sys.path.append('/mnt/skin_cancer_nas/data/torch_generator')
from skin_cancer_nas.data.torch_generator import generator as data_gen
from skin_cancer_nas.data.torch_generator import base_classes
from skin_cancer_nas.data.torch_generator import config
from skin_cancer_nas.data.torch_generator import preprocessor

from base_classes import Dataset
from torchvision import transforms

device = 'cpu'

if __name__ == "__main__":
    # 1 Let's see data
    partition, labels = data_gen.train_val_split(val_size=0.1)

    MEAN = [0.2336, 0.6011, 0.3576, 0.4543]
    STD = [0.0530, 0.0998, 0.0965, 0.1170]
    normalize = [
        transforms.Normalize(MEAN, STD),

    ]
    train_transform = transforms.Compose(normalize)
    valid_transform = transforms.Compose(normalize)
    
    # Generators Declaration
    training_set = Dataset(partition['train'], labels, 
                           transform=train_transform, 
                           device=device)
    training_generator = torch.utils.data.DataLoader(training_set, **data_gen.PARAMS, pin_memory=True)
    validation_set = Dataset(partition['validation'], labels, 
                           transform=valid_transform, 
                           device=device)
    validation_generator = torch.utils.data.DataLoader(validation_set, **data_gen.PARAMS, pin_memory=True)

    
    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

