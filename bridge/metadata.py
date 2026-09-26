"""Bounded PNG/JPEG/WebP metadata reader. Does not decode image pixels."""
from __future__ import annotations
import struct
import zlib
from .spec import BridgeError, MAX_METADATA

MAX_IMAGE = 128 * 1024 * 1024


def image_dimensions(data: bytes):
    """Read the container's output size without decoding pixels."""
    from io import BytesIO
    from PIL import Image
    if len(data) > MAX_IMAGE:
        raise BridgeError('IMPORT_LIMIT_EXCEEDED', 'Image exceeds 128 MiB')
    try:
        with Image.open(BytesIO(data)) as image:
            return image.size
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise BridgeError('INVALID_METADATA', 'Cannot read source image dimensions') from exc


def _text(data: bytes, encoding='utf-8') -> str:
    try: return data.decode(encoding).rstrip('\x00')
    except UnicodeError as exc: raise BridgeError('INVALID_METADATA', f'Invalid {encoding} metadata') from exc


def _inflate(data: bytes) -> bytes:
    try:
        obj = zlib.decompressobj()
        out = obj.decompress(data, MAX_METADATA + 1)
        if len(out) > MAX_METADATA or obj.unconsumed_tail or not obj.eof:
            raise BridgeError('IMPORT_LIMIT_EXCEEDED', 'Compressed metadata is too large or incomplete')
        return out
    except zlib.error as exc: raise BridgeError('INVALID_METADATA', 'Invalid compressed metadata') from exc


class Collector:
    def __init__(self): self.values, self.total = {}, 0
    def add(self, key, value):
        self.total += len(key.encode('utf-8')) + len(value.encode('utf-8'))
        if self.total > MAX_METADATA: raise BridgeError('IMPORT_LIMIT_EXCEEDED', 'Expanded metadata exceeds 16 MiB')
        k = key.casefold()
        if k in self.values and self.values[k] != value:
            raise BridgeError('INVALID_METADATA', f'Conflicting metadata field {key}')
        self.values[k] = value
    def tagged(self, value):
        key, sep, rest = value.partition(':')
        if sep and key.lower() in ('workflow','prompt','parameters'):
            self.add(key, rest)
        elif '\nSteps:' in value:
            self.add('parameters', value)


def _tiff(data: bytes, out: Collector):
    if data.startswith(b'Exif\0\0'): data = data[6:]
    if len(data) < 8 or data[:2] not in (b'II', b'MM'): raise BridgeError('INVALID_METADATA', 'Invalid TIFF header')
    endian = '<' if data[:2] == b'II' else '>'
    def read(offset, fmt):
        size = struct.calcsize(fmt)
        if offset < 0 or offset + size > len(data): raise BridgeError('INVALID_METADATA', 'EXIF offset out of range')
        return struct.unpack_from(endian + fmt, data, offset)[0]
    if read(2, 'H') != 42: raise BridgeError('INVALID_METADATA', 'Invalid TIFF signature')
    visited = set()
    def walk(offset, depth, follow_next=True):
        if offset == 0: return
        if depth > 8 or offset in visited: raise BridgeError('INVALID_METADATA', 'EXIF recursion/cycle limit')
        visited.add(offset)
        count = read(offset, 'H')
        if count > 4096 or offset + 2 + 12 * count + (4 if follow_next else 0) > len(data): raise BridgeError('INVALID_METADATA', 'Invalid IFD table size')
        for i in range(count):
            entry = offset + 2 + 12 * i
            tag, typ, n = read(entry, 'H'), read(entry+2, 'H'), read(entry+4, 'I')
            size = {1:1,2:1,3:2,4:4,5:8,7:1,9:4,10:8,11:4,12:8}.get(typ)
            if size is None: continue
            length = n * size
            if length > MAX_METADATA: raise BridgeError('IMPORT_LIMIT_EXCEEDED', 'EXIF value exceeds metadata limit')
            value_offset = entry+8 if length <= 4 else read(entry+8, 'I')
            if value_offset + length > len(data): raise BridgeError('INVALID_METADATA', 'EXIF value out of range')
            raw = data[value_offset:value_offset+length]
            if tag in (0x8769,0xa005) and typ == 4 and n == 1:
                walk(read(entry+8, 'I'), depth+1, False)
            elif tag == 0x9286 and typ in (2,7):
                header, body = raw[:8], raw[8:]
                if header.startswith(b'UNICODE'):
                    enc = 'utf-16' if body.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-16-be'
                    value = _text(body, enc)
                elif header.startswith(b'ASCII'):
                    value = _text(body, 'ascii')
                elif raw.startswith(b'\xef\xbb\xbf'):
                    value = _text(raw, 'utf-8-sig')
                elif header.startswith(b'JIS'):
                    raise BridgeError('INVALID_METADATA', 'JIS UserComment is not supported')
                else:
                    value = _text(raw)
                out.tagged(value)
            elif typ == 2:
                out.tagged(_text(raw))
        if follow_next: walk(read(offset+2+12*count, 'I'), depth+1)
    walk(read(4,'I'),0)


def read_metadata(data: bytes) -> dict:
    if len(data) > MAX_IMAGE: raise BridgeError('IMPORT_LIMIT_EXCEEDED', 'Image exceeds 128 MiB')
    out = Collector()
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        pos = 8
        while pos < len(data):
            if pos + 12 > len(data): raise BridgeError('INVALID_METADATA', 'Truncated PNG chunk')
            n = int.from_bytes(data[pos:pos+4], 'big'); kind = data[pos+4:pos+8]
            end = pos+12+n
            if end > len(data): raise BridgeError('INVALID_METADATA', 'PNG chunk out of range')
            raw = data[pos+8:pos+8+n]
            if kind in (b'tEXt',b'zTXt',b'iTXt',b'eXIf'):
                if n > MAX_METADATA: raise BridgeError('IMPORT_LIMIT_EXCEEDED', 'PNG metadata too large')
                expected = int.from_bytes(data[pos+8+n:end], 'big')
                if zlib.crc32(kind+raw) & 0xffffffff != expected: raise BridgeError('INVALID_METADATA', 'PNG metadata CRC mismatch')
                if kind == b'eXIf': _tiff(raw,out)
                elif kind == b'tEXt':
                    key, sep, val = raw.partition(b'\0')
                    if not sep: raise BridgeError('INVALID_METADATA', 'Missing PNG keyword separator')
                    out.add(_text(key,'latin1'), _text(val, 'latin1'))
                elif kind == b'zTXt':
                    key, sep, val = raw.partition(b'\0')
                    if not sep or not val or val[0] != 0: raise BridgeError('INVALID_METADATA', 'Invalid zTXt')
                    out.add(_text(key,'latin1'), _text(_inflate(val[1:]),'latin1'))
                else:
                    key, sep, val = raw.partition(b'\0')
                    if not sep or len(val)<2 or val[0] not in (0,1) or val[1] != 0: raise BridgeError('INVALID_METADATA', 'Invalid iTXt')
                    compressed = val[0];parts=val[2:].split(b'\0',2)
                    if len(parts)!=3: raise BridgeError('INVALID_METADATA', 'Invalid iTXt language fields')
                    out.add(_text(key,'latin1'), _text(_inflate(parts[2]) if compressed else parts[2]))
            pos=end
            if kind == b'IEND': break
    elif data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        if len(data) < 12: raise BridgeError('INVALID_METADATA', 'Truncated WebP')
        limit = int.from_bytes(data[4:8],'little')+8
        if limit != len(data): raise BridgeError('INVALID_METADATA','Invalid RIFF size')
        pos=12
        while pos < limit:
            if pos+8>limit: raise BridgeError('INVALID_METADATA','Truncated RIFF chunk')
            kind=data[pos:pos+4];n=int.from_bytes(data[pos+4:pos+8],'little')
            end=pos+8+n
            if end+(n%2)>limit: raise BridgeError('INVALID_METADATA','RIFF chunk out of range')
            if kind == b'EXIF':
                if n>MAX_METADATA: raise BridgeError('IMPORT_LIMIT_EXCEEDED','EXIF too large')
                _tiff(data[pos+8:end],out)
            pos=end+(n%2)
    elif data.startswith(b'\xff\xd8'):
        pos=2
        while pos<len(data):
            if data[pos]!=255: raise BridgeError('INVALID_METADATA','Invalid JPEG marker')
            while pos<len(data) and data[pos]==255:pos+=1
            if pos>=len(data):break
            kind=data[pos];pos+=1
            if kind in (0xda,0xd9):break
            if kind in (0xd8,0x01) or 0xd0<=kind<=0xd7:continue
            if pos+2>len(data):raise BridgeError('INVALID_METADATA','Truncated JPEG')
            n=int.from_bytes(data[pos:pos+2],'big')
            if n<2 or pos+n>len(data):raise BridgeError('INVALID_METADATA','JPEG segment out of range')
            raw=data[pos+2:pos+n]
            if kind==0xe1 and raw.startswith(b'Exif\0\0'):_tiff(raw,out)
            if kind==0xfe:out.tagged(_text(raw))
            pos+=n
    else:
        raise BridgeError('NOT_SUPPORTED_IMAGE','Expected a PNG, JPEG or WebP image file')
    if 'workflow' in out.values or 'prompt' in out.values:
        return {'kind':'native','metadata':out.values}
    return {'kind':'forge' if '\nSteps:' in out.values.get('parameters','') else 'unknown','metadata':out.values}
