# Writes tests/data/tiny.pth: a minimal PyTorch zip checkpoint (no torch needed) with
#   "plain"      float32 [2, 3], contiguous, values 0..5
#   "strided"    float32 [3, 2], the transpose of "plain"'s storage (stride [1, 3]), like Meta's pos_embed
#   "offset"     float32 [2], a view at element offset 4 into a shared 6-element storage (stored in bytes by candle)
#   "counter"    int64 [] (batch norm's num_batches_tracked), which the converter must skip
# Run: python -I tests/data/make_tiny_pth.py tests/data/tiny.pth
import collections, io, pickle, struct, sys, types, zipfile

torch = types.ModuleType("torch")
utils = types.ModuleType("torch._utils")
sys.modules["torch"] = torch
sys.modules["torch._utils"] = utils


def _rebuild_tensor_v2(*args):
    raise RuntimeError("placeholder")


_rebuild_tensor_v2.__module__ = "torch._utils"
utils._rebuild_tensor_v2 = _rebuild_tensor_v2


class FloatStorage:
    pass


class LongStorage:
    pass


for c in (FloatStorage, LongStorage):
    c.__module__ = "torch"
    setattr(torch, c.__name__, c)


class Storage:
    def __init__(self, cls, key, n):
        self.cls, self.key, self.n = cls, key, n


class Tensor:
    def __init__(self, storage, offset, size, stride):
        self.args = (storage, offset, tuple(size), tuple(stride), False, collections.OrderedDict())

    def __reduce__(self):
        return (_rebuild_tensor_v2, self.args)


class Pickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, Storage):
            return ("storage", obj.cls, obj.key, "cpu", obj.n)
        return None


floats = Storage(FloatStorage, "0", 6)
longs = Storage(LongStorage, "1", 1)
state = {
    "plain": Tensor(floats, 0, [2, 3], [3, 1]),
    "strided": Tensor(floats, 0, [3, 2], [1, 3]),
    "offset": Tensor(floats, 4, [2], [1]),
    "counter": Tensor(longs, 0, [], []),
}
buf = io.BytesIO()
Pickler(buf, protocol=2).dump(state)
out = sys.argv[1]
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as z:
    z.writestr("tiny/data.pkl", buf.getvalue())
    z.writestr("tiny/data/0", struct.pack("<6f", 0, 1, 2, 3, 4, 5))
    z.writestr("tiny/data/1", struct.pack("<q", 7))
    z.writestr("tiny/version", "3\n")
print(out)
