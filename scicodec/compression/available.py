


available_compressors = {
    'zlib':{"bounds":["level"],"package":"numcodecs"},
    'gzip':{"bounds":["level"],"package":"numcodecs"},
    'bz2':{"bounds":["level"],"package":"numcodecs"},
    'lzma':{"bounds":["level"],"package":"numcodecs"},
    'lz4':{"bounds":["level"],"package":"numcodecs"},
    'blosc':{"bounds":["level"],"package":"numcodecs"},
    'fpzip':{"bounds":["rel_precision"],"package":"numcodecs"},
    'zstd':{"bounds":["level"],"package":"numcodecs"},
    'sz3':{"bounds":["abs_precision"],"package":"libpressio"},
    'mgard':{"bounds":["abs_precision"],"package":"libpressio"},
    'zfp':{"bounds":["fixed_ratio","bit_precision"],"package":"libpressio"},
    'mbt2018':{"bounds":["level","device"],"package":"compressai"},
    'cheng2020_anchor':{"bounds":["level","device"],"package":"compressai"},
    'bmshj2018_factorized':{"bounds":["level","device"],"package":"compressai"},

}

supported_error_bounds = {
    compressor: bounds["bounds"] for compressor, bounds in available_compressors.items()
}
supported_compressors = list(available_compressors.keys())
