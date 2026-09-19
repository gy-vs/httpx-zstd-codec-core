from __future__ import annotations

import struct
import subprocess
import sys
import typing
import zlib

import chardet
import pytest

import httpx


def test_deflate():
    """
    Deflate encoding may use either 'zlib' or 'deflate' in the wild.

    https://stackoverflow.com/questions/1838699/how-can-i-decompress-a-gzip-stream-with-zlib#answer-22311297
    """
    body = b"test 123"
    compressor = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    compressed_body = compressor.compress(body) + compressor.flush()

    headers = [(b"Content-Encoding", b"deflate")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body


def test_zlib():
    """
    Deflate encoding may use either 'zlib' or 'deflate' in the wild.

    https://stackoverflow.com/questions/1838699/how-can-i-decompress-a-gzip-stream-with-zlib#answer-22311297
    """
    body = b"test 123"
    compressed_body = zlib.compress(body)

    headers = [(b"Content-Encoding", b"deflate")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body


def test_gzip():
    body = b"test 123"
    compressor = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS | 16)
    compressed_body = compressor.compress(body) + compressor.flush()

    headers = [(b"Content-Encoding", b"gzip")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body


def test_brotli():
    body = b"test 123"
    compressed_body = b"\x8b\x03\x80test 123\x03"

    headers = [(b"Content-Encoding", b"br")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body


def test_zstd():
    zstandard = pytest.importorskip("zstandard")

    body = b"test 123"
    compressed_body = zstandard.ZstdCompressor().compress(body)

    headers = [(b"Content-Encoding", b"zstd")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body


def test_zstd_multiple_frames():
    zstandard = pytest.importorskip("zstandard")

    body = b"test 123"
    compressed_body = zstandard.ZstdCompressor().compress(body)

    headers = [(b"Content-Encoding", b"zstd")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body + compressed_body,
    )
    assert response.content == body + body


def test_zstd_skippable_frame():
    zstandard = pytest.importorskip("zstandard")

    body = b"test 123"
    compressed_body = zstandard.ZstdCompressor().compress(body)
    # See: https://www.rfc-editor.org/rfc/rfc8878#name-skippable-frames
    skippable_payload = b"skippable"
    skippable_frame = (
        struct.pack("<I", 0x184D2A50)
        + struct.pack("<I", len(skippable_payload))
        + skippable_payload
    )

    headers = [(b"Content-Encoding", b"zstd")]

    # A skippable frame may appear before the zstd frames.
    response = httpx.Response(
        200, headers=headers, content=skippable_frame + compressed_body
    )
    assert response.content == body

    # A skippable frame may appear between zstd frames.
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body + skippable_frame + compressed_body,
    )
    assert response.content == body + body

    # A skippable frame may appear after the zstd frames.
    response = httpx.Response(
        200, headers=headers, content=compressed_body + skippable_frame
    )
    assert response.content == body

    # A stream containing only skippable frames decodes to empty content.
    response = httpx.Response(
        200,
        headers=headers,
        content=skippable_frame + skippable_frame,
    )
    assert response.content == b""


def test_zstd_skippable_frame_split_across_chunks():
    zstandard = pytest.importorskip("zstandard")
    from httpx._decoders import ZStandardDecoder

    body = b"test 123"
    compressed_body = zstandard.ZstdCompressor().compress(body)
    skippable_payload = b"skippable"
    skippable_frame = (
        struct.pack("<I", 0x184D2A50)
        + struct.pack("<I", len(skippable_payload))
        + skippable_payload
    )

    # The skippable frame header itself spans two decode calls.
    decoder = ZStandardDecoder()
    decoded = (
        decoder.decode(skippable_frame[:3])
        + decoder.decode(skippable_frame[3:] + compressed_body)
        + decoder.flush()
    )
    assert decoded == body


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 8, 16, 128])
def test_zstd_streaming_chunk_boundaries(chunk_size):
    zstandard = pytest.importorskip("zstandard")

    body = b"test 123" * 100
    compressed_body = zstandard.ZstdCompressor().compress(body)
    skippable_payload = b"skippable"
    skippable_frame = (
        struct.pack("<I", 0x184D2A50)
        + struct.pack("<I", len(skippable_payload))
        + skippable_payload
    )
    compressed = compressed_body + skippable_frame + compressed_body
    chunks = [
        compressed[i : i + chunk_size]
        for i in range(0, len(compressed), chunk_size)
    ]

    headers = [(b"Content-Encoding", b"zstd")]

    response = httpx.Response(200, headers=headers, content=iter(chunks))
    assert b"".join(response.iter_bytes()) == body + body
    assert response.num_bytes_downloaded == len(compressed)


@pytest.mark.anyio
async def test_zstd_async_streaming():
    zstandard = pytest.importorskip("zstandard")

    body = b"test 123" * 100
    compressed_body = zstandard.ZstdCompressor().compress(body)
    skippable_payload = b"skippable"
    skippable_frame = (
        struct.pack("<I", 0x184D2A50)
        + struct.pack("<I", len(skippable_payload))
        + skippable_payload
    )
    compressed = compressed_body + skippable_frame + compressed_body
    chunks = [compressed[i : i + 7] for i in range(0, len(compressed), 7)]

    async def compress() -> typing.AsyncIterator[bytes]:
        for chunk in chunks:
            yield chunk

    headers = [(b"Content-Encoding", b"zstd")]
    response = httpx.Response(200, headers=headers, content=compress())
    assert not hasattr(response, "body")
    assert await response.aread() == body + body
    assert response.num_bytes_downloaded == len(compressed)


def test_zstd_empty_content():
    pytest.importorskip("zstandard")

    headers = [(b"Content-Encoding", b"zstd")]
    response = httpx.Response(200, headers=headers, content=b"")
    assert response.content == b""


def test_zstd_corrupt_data():
    pytest.importorskip("zstandard")

    headers = [(b"Content-Encoding", b"zstd")]
    with pytest.raises(httpx.DecodingError):
        httpx.Response(200, headers=headers, content=b"invalid")


def test_zstd_truncated_frame():
    zstandard = pytest.importorskip("zstandard")

    compressed_body = zstandard.ZstdCompressor().compress(b"test 123" * 100)

    headers = [(b"Content-Encoding", b"zstd")]
    with pytest.raises(httpx.DecodingError):
        httpx.Response(
            200, headers=headers, content=compressed_body[:10]
        )

    # A truncated frame in a streaming response raises on completion.
    response = httpx.Response(
        200,
        headers=headers,
        content=[compressed_body[:10]],
    )
    with pytest.raises(httpx.DecodingError):
        b"".join(response.iter_bytes())


def test_multi():
    body = b"test 123"

    deflate_compressor = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    compressed_body = deflate_compressor.compress(body) + deflate_compressor.flush()

    gzip_compressor = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS | 16)
    compressed_body = (
        gzip_compressor.compress(compressed_body) + gzip_compressor.flush()
    )

    headers = [(b"Content-Encoding", b"deflate, gzip")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body


def test_multi_with_zstd():
    zstandard = pytest.importorskip("zstandard")

    body = b"test 123"
    compressed_body = zstandard.ZstdCompressor().compress(body)

    # `zstd, gzip` means zstd was applied first, then gzip.
    gzip_compressor = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS | 16)
    compressed_body = (
        gzip_compressor.compress(compressed_body) + gzip_compressor.flush()
    )

    headers = [(b"Content-Encoding", b"zstd, gzip")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body

    # `gzip, zstd` means gzip was applied first, then zstd.
    gzip_compressor = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS | 16)
    gzip_body = gzip_compressor.compress(body) + gzip_compressor.flush()
    compressed_body = zstandard.ZstdCompressor().compress(gzip_body)

    headers = [(b"Content-Encoding", b"gzip, zstd")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body


def test_multi_with_identity():
    body = b"test 123"
    compressed_body = b"\x8b\x03\x80test 123\x03"

    headers = [(b"Content-Encoding", b"br, identity")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body

    headers = [(b"Content-Encoding", b"identity, br")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body


@pytest.mark.anyio
async def test_streaming():
    body = b"test 123"
    compressor = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS | 16)

    async def compress(body: bytes) -> typing.AsyncIterator[bytes]:
        yield compressor.compress(body)
        yield compressor.flush()

    headers = [(b"Content-Encoding", b"gzip")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compress(body),
    )
    assert not hasattr(response, "body")
    assert await response.aread() == body


@pytest.mark.parametrize(
    "header_value", (b"deflate", b"gzip", b"br", b"zstd", b"identity")
)
def test_empty_content(header_value):
    headers = [(b"Content-Encoding", header_value)]
    response = httpx.Response(
        200,
        headers=headers,
        content=b"",
    )
    assert response.content == b""


@pytest.mark.parametrize(
    "header_value", (b"deflate", b"gzip", b"br", b"zstd", b"identity")
)
def test_decoders_empty_cases(header_value):
    headers = [(b"Content-Encoding", header_value)]
    response = httpx.Response(content=b"", status_code=200, headers=headers)
    assert response.read() == b""


def test_zstd_decoder_requires_zstandard(monkeypatch):
    # Simulate the optional 'zstandard' package not being installed.
    import httpx._decoders as decoders

    monkeypatch.setattr(decoders, "zstandard", None)
    with pytest.raises(ImportError, match="httpx\\[zstd\\]"):
        decoders.ZStandardDecoder()


def test_zstd_decoder_registration():
    import httpx._decoders as decoders

    try:
        import zstandard  # noqa: F401
    except ImportError:
        assert "zstd" not in decoders.SUPPORTED_DECODERS
    else:
        assert "zstd" in decoders.SUPPORTED_DECODERS


def test_zstd_accept_encoding():
    # The default Accept-Encoding header must agree with the installed
    # content decoders.
    from httpx._client import ACCEPT_ENCODING

    assert ("zstd" in ACCEPT_ENCODING) == (
        "zstd" in httpx._decoders.SUPPORTED_DECODERS
    )


def test_zstd_without_optional_dependency():
    """
    Importing httpx must not fail when 'zstandard' is not installed,
    and zstd must not be advertised in the default Accept-Encoding.
    """
    code = """
import sys
import builtins

real_import = builtins.__import__


def blocked_import(name, *args, **kwargs):
    if name == "zstandard" or name.startswith("zstandard."):
        raise ImportError("No module named 'zstandard'")
    return real_import(name, *args, **kwargs)


builtins.__import__ = blocked_import

import httpx

assert httpx._compat.zstandard is None
assert "zstd" not in httpx._decoders.SUPPORTED_DECODERS
from httpx._client import ACCEPT_ENCODING

assert "zstd" not in ACCEPT_ENCODING

# Instantiating the decoder fails with an informative error,
# but responses with an unsupported encoding pass through.
try:
    httpx._decoders.ZStandardDecoder()
except ImportError:
    pass
else:
    raise AssertionError("ZStandardDecoder() should raise ImportError")

response = httpx.Response(
    200, headers=[("Content-Encoding", "zstd")], content=b"raw"
)
assert response.content == b"raw"
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("header_value", (b"deflate", b"gzip", b"br", b"zstd"))
def test_decoding_errors(header_value):
    headers = [(b"Content-Encoding", header_value)]
    compressed_body = b"invalid"
    with pytest.raises(httpx.DecodingError):
        request = httpx.Request("GET", "https://example.org")
        httpx.Response(200, headers=headers, content=compressed_body, request=request)

    with pytest.raises(httpx.DecodingError):
        httpx.Response(200, headers=headers, content=compressed_body)


@pytest.mark.parametrize(
    ["data", "encoding"],
    [
        ((b"Hello,", b" world!"), "ascii"),
        ((b"\xe3\x83", b"\x88\xe3\x83\xa9", b"\xe3", b"\x83\x99\xe3\x83\xab"), "utf-8"),
        ((b"Euro character: \x88! abcdefghijklmnopqrstuvwxyz", b""), "cp1252"),
        ((b"Accented: \xd6sterreich abcdefghijklmnopqrstuvwxyz", b""), "iso-8859-1"),
    ],
)
@pytest.mark.anyio
async def test_text_decoder_with_autodetect(data, encoding):
    async def iterator() -> typing.AsyncIterator[bytes]:
        nonlocal data
        for chunk in data:
            yield chunk

    def autodetect(content):
        return chardet.detect(content).get("encoding")

    # Accessing `.text` on a read response.
    response = httpx.Response(200, content=iterator(), default_encoding=autodetect)
    await response.aread()
    assert response.text == (b"".join(data)).decode(encoding)

    # Streaming `.aiter_text` iteratively.
    # Note that if we streamed the text *without* having read it first, then
    # we won't get a `charset_normalizer` guess, and will instead always rely
    # on utf-8 if no charset is specified.
    text = "".join([part async for part in response.aiter_text()])
    assert text == (b"".join(data)).decode(encoding)


@pytest.mark.anyio
async def test_text_decoder_known_encoding():
    async def iterator() -> typing.AsyncIterator[bytes]:
        yield b"\x83g"
        yield b"\x83"
        yield b"\x89\x83x\x83\x8b"

    response = httpx.Response(
        200,
        headers=[(b"Content-Type", b"text/html; charset=shift-jis")],
        content=iterator(),
    )

    await response.aread()
    assert "".join(response.text) == "トラベル"


def test_text_decoder_empty_cases():
    response = httpx.Response(200, content=b"")
    assert response.text == ""

    response = httpx.Response(200, content=[b""])
    response.read()
    assert response.text == ""


@pytest.mark.parametrize(
    ["data", "expected"],
    [((b"Hello,", b" world!"), ["Hello,", " world!"])],
)
def test_streaming_text_decoder(
    data: typing.Iterable[bytes], expected: list[str]
) -> None:
    response = httpx.Response(200, content=iter(data))
    assert list(response.iter_text()) == expected


def test_line_decoder_nl():
    response = httpx.Response(200, content=[b""])
    assert list(response.iter_lines()) == []

    response = httpx.Response(200, content=[b"", b"a\n\nb\nc"])
    assert list(response.iter_lines()) == ["a", "", "b", "c"]

    # Issue #1033
    response = httpx.Response(
        200, content=[b"", b"12345\n", b"foo ", b"bar ", b"baz\n"]
    )
    assert list(response.iter_lines()) == ["12345", "foo bar baz"]


def test_line_decoder_cr():
    response = httpx.Response(200, content=[b"", b"a\r\rb\rc"])
    assert list(response.iter_lines()) == ["a", "", "b", "c"]

    response = httpx.Response(200, content=[b"", b"a\r\rb\rc\r"])
    assert list(response.iter_lines()) == ["a", "", "b", "c"]

    # Issue #1033
    response = httpx.Response(
        200, content=[b"", b"12345\r", b"foo ", b"bar ", b"baz\r"]
    )
    assert list(response.iter_lines()) == ["12345", "foo bar baz"]


def test_line_decoder_crnl():
    response = httpx.Response(200, content=[b"", b"a\r\n\r\nb\r\nc"])
    assert list(response.iter_lines()) == ["a", "", "b", "c"]

    response = httpx.Response(200, content=[b"", b"a\r\n\r\nb\r\nc\r\n"])
    assert list(response.iter_lines()) == ["a", "", "b", "c"]

    response = httpx.Response(200, content=[b"", b"a\r", b"\n\r\nb\r\nc"])
    assert list(response.iter_lines()) == ["a", "", "b", "c"]

    # Issue #1033
    response = httpx.Response(200, content=[b"", b"12345\r\n", b"foo bar baz\r\n"])
    assert list(response.iter_lines()) == ["12345", "foo bar baz"]


def test_invalid_content_encoding_header():
    headers = [(b"Content-Encoding", b"invalid-header")]
    body = b"test 123"

    response = httpx.Response(
        200,
        headers=headers,
        content=body,
    )
    assert response.content == body
