from timm.models import create_model  
from .MetaFG import *
from .MetaFG_meta import *
def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'MetaFG':
        model = create_model(
                config.MODEL.NAME,
                pretrained=False,
                num_classes=config.MODEL.NUM_CLASSES, 
                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                img_size=config.DATA.IMG_SIZE,
                only_last_cls=config.MODEL.ONLY_LAST_CLS,
                extra_token_num=config.MODEL.EXTRA_TOKEN_NUM,
                meta_dims=config.MODEL.META_DIMS
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
