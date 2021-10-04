from models.config_space.binary_fcn_cs import BinaryFCN
from models.config_space.binary_inception import BinaryInception
from models.config_space.binary_mlp_cs import BinaryMLP
from models.config_space.binary_quick_net_cs import BinaryQuickNet
from models.config_space.binary_resnet import BinaryResNet

BINARY_CLASSIFIER_NAMES = [
    'BINARY_QUICKNET',
    'BINARY_INCEPTION',
    'BINARY_FCN',
    'BINARY_MLP',
    'BINARY_RESNET',
]


def load_model(classifier_name, input_shape, num_classes):
    if classifier_name not in BINARY_CLASSIFIER_NAMES:
        raise RuntimeError(f'Name {classifier_name} invalid.')

    if classifier_name == 'BINARY_MLP':
        return BinaryMLP(input_shape, num_classes)
    elif classifier_name == 'BINARY_FCN':
        return BinaryFCN(input_shape, num_classes)
    elif classifier_name == 'BINARY_INCEPTION':
        return BinaryInception(input_shape, num_classes)
    elif classifier_name == 'BINARY_QUICKNET':
        return BinaryQuickNet(input_shape, num_classes)
    elif classifier_name == 'BINARY_RESNET':
        return BinaryResNet(input_shape, num_classes)

