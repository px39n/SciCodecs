from compressai.zoo import bmshj2018_factorized as bmshj2018_factorized_base

def bmshj2018_factorized(quality, pretrained=True, **kwargs):
    model = bmshj2018_factorized_base(quality=quality, pretrained=pretrained, **kwargs)
    model.codec_id = "bmshj2018"
    return model
