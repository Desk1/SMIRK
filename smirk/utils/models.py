def get_test_model_name(target_model_name: str):
    """
    Map of suitable test models for given target models
    """
    match target_model_name:
        case 'vgg16bn':
            return 'vgg16'
        case 'vgg16':
            return 'vgg16bn'
        case 'resnet50':
            return 'inception_resnetv1_vggface2'
        case 'inception_resnetv1_vggface2':
            return 'resnet50'
        case 'mobilenet_v2':
            return 'resnet50'
        case 'efficientnet_b0':
            return 'resnet50'
        case 'inception_v3':
            return 'resnet50'
        case 'swin_transformer':
            return 'resnet50'
        case 'vision_transformer':
            return 'resnet50'
        case 'inception_resnetv1_casia':
            return 'efficientnet_b0_casia'
        case 'efficientnet_b0_casia':
            return 'inception_resnetv1_casia'
        case 'sphere20a':
            return 'inception_resnetv1_casia'
        case _:
            return 'resnet50'   