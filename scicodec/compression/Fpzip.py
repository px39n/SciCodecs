
import xarray as xr
import zarr
import fpzip
from numcodecs.abc import Codec
from numcodecs import register_codec, Zstd
from numcodecs.compat import ndarray_copy, ensure_contiguous_ndarray, ensure_bytes

class FPZip(Codec):
    codec_id = 'fpzip'
    def __init__(self, precision=0):
        self.precision = precision
        
    def encode(self, buf):
        return fpzip.compress(buf, precision=self.precision, order='C')

    def decode(self, buf, out=None):

        buf = ensure_bytes(buf)
        if out is not None:
            out = ensure_contiguous_ndarray(out)

        # do decompression
        dec =fpzip.decompress(buf, order='C')

        # handle destination
        if out is not None:
            return ndarray_copy(dec, out)
        else:
            return dec

from numcodecs.registry import register_codec
register_codec(FPZip)
 