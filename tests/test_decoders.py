from __future__ import annotations

import io
import pathlib
import subprocess
import sys
import typing
import zlib

import chardet
import pytest
import zstandard as zstd

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
    body = b"test 123"
    compressed_body = zstd.compress(body)

    headers = [(b"Content-Encoding", b"zstd")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body


def test_zstd_decoding_error():
    compressed_body = "this_is_not_zstd_compressed_data"

    headers = [(b"Content-Encoding", b"zstd")]
    with pytest.raises(httpx.DecodingError):
        httpx.Response(
            200,
            headers=headers,
            content=compressed_body,
        )


def test_zstd_empty():
    headers = [(b"Content-Encoding", b"zstd")]
    response = httpx.Response(200, headers=headers, content=b"")
    assert response.content == b""


def test_zstd_truncated():
    body = b"test 123"
    compressed_body = zstd.compress(body)

    headers = [(b"Content-Encoding", b"zstd")]
    with pytest.raises(httpx.DecodingError):
        httpx.Response(
            200,
            headers=headers,
            content=compressed_body[1:3],
        )


def test_zstd_multiframe():
    # test inspired by urllib3 test suite
    data = (
        # Zstandard frame
        zstd.compress(b"foo")
        # skippable frame (must be ignored)
        + bytes.fromhex(
            "50 2A 4D 18"  # Magic_Number (little-endian)
            "07 00 00 00"  # Frame_Size (little-endian)
            "00 00 00 00 00 00 00"  # User_Data
        )
        # Zstandard frame
        + zstd.compress(b"bar")
    )
    compressed_body = io.BytesIO(data)

    headers = [(b"Content-Encoding", b"zstd")]
    response = httpx.Response(200, headers=headers, content=compressed_body)
    response.read()
    assert response.content == b"foobar"


def test_zstd_multiple_frames():
    body = b"test 123 " * 1000
    compressor = zstd.ZstdCompressor()
    compressed_body = compressor.compress(body[:5000]) + compressor.compress(
        body[5000:]
    )

    headers = [(b"Content-Encoding", b"zstd")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body


def test_zstd_skippable_frames():
    body = b"test 123"
    compressed_body = zstd.compress(body)

    def skippable_frame(payload: bytes) -> bytes:
        # Skippable frames use a Magic_Number in the range 0x184D2A50-0x184D2A5F,
        # followed by a little-endian Frame_Size and the User_Data.
        return (
            (0x184D2A50).to_bytes(4, "little")
            + len(payload).to_bytes(4, "little")
            + payload
        )

    data = (
        skippable_frame(b"before")
        + compressed_body
        + skippable_frame(b"")
        + skippable_frame(b"after")
    )

    headers = [(b"Content-Encoding", b"zstd")]
    response = httpx.Response(200, headers=headers, content=data)
    assert response.content == body

    # A stream made up of skippable frames only decodes to empty content.
    response = httpx.Response(
        200, headers=headers, content=skippable_frame(b"nothing here")
    )
    assert response.content == b""


@pytest.mark.parametrize("chunk_size", range(1, 12))
def test_zstd_streaming_chunk_boundaries(chunk_size):
    """
    Data may be split across chunk boundaries at *any* point in the stream,
    including inside frame and skippable frame headers.
    """
    compressed_frame = zstd.compress(b"Hello, ")
    skippable_frame = bytes.fromhex(
        "50 2A 4D 18"  # Magic_Number (little-endian)
        "07 00 00 00"  # Frame_Size (little-endian)
        "00 00 00 00 00 00 00"  # User_Data
    )
    data = compressed_frame + skippable_frame + zstd.compress(b"world!")

    chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

    headers = [(b"Content-Encoding", b"zstd")]
    response = httpx.Response(200, headers=headers, content=iter(chunks))
    assert b"".join(response.iter_bytes()) == b"Hello, world!"
    assert response.num_bytes_downloaded == len(data)


@pytest.mark.anyio
async def test_zstd_async_streaming_chunk_boundaries():
    body = b"test 123 " * 100
    compressed_body = zstd.compress(body)

    async def iterator() -> typing.AsyncIterator[bytes]:
        yield compressed_body[:5]
        yield b""
        yield compressed_body[5:20]
        yield compressed_body[20:]

    headers = [(b"Content-Encoding", b"zstd")]
    response = httpx.Response(200, headers=headers, content=iterator())
    assert await response.aread() == body
    assert response.num_bytes_downloaded == len(compressed_body)


@pytest.mark.parametrize(
    "invalid_body",
    [
        pytest.param(b"invalid", id="garbage"),
        pytest.param(zstd.compress(b"test 123")[:-3], id="truncated-frame"),
        # A declared skippable frame whose payload is cut short.
        pytest.param(
            (0x184D2A50).to_bytes(4, "little")
            + (100).to_bytes(4, "little")
            + b"too-short",
            id="truncated-skippable-frame",
        ),
        # A complete frame followed by the start of a second frame.
        pytest.param(
            zstd.compress(b"test 123") + b"\x28\xb5\x2f\xfd", id="partial-next-frame"
        ),
    ],
)
def test_zstd_errors(invalid_body):
    headers = [(b"Content-Encoding", b"zstd")]
    request = httpx.Request("GET", "https://example.org")

    # The error is surfaced both for buffered content...
    with pytest.raises(httpx.DecodingError):
        httpx.Response(200, headers=headers, content=invalid_body, request=request)

    # ...and while iterating over a streamed response.
    response = httpx.Response(
        200, headers=headers, content=iter([invalid_body]), request=request
    )
    with pytest.raises(httpx.DecodingError):
        b"".join(response.iter_bytes())


def test_zstd_combined_encoding():
    """
    Multiple encodings are peeled off in the reverse of the order given
    in the Content-Encoding header.
    """
    body = b"test 123 " * 100

    gzip_compressor = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS | 16)
    gzipped = gzip_compressor.compress(body) + gzip_compressor.flush()

    # `gzip, zstd` means gzip was applied first, then zstd.
    compressed_body = zstd.compress(gzipped)

    headers = [(b"Content-Encoding", b"gzip, zstd")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body

    # The same should hold when the data is streamed in arbitrary chunks.
    response = httpx.Response(
        200,
        headers=headers,
        content=iter([compressed_body[:7], b"", compressed_body[7:]]),
    )
    assert b"".join(response.iter_bytes()) == body
    assert response.num_bytes_downloaded == len(compressed_body)


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


def test_zstd_optional_dependency_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When `zstandard` is installed, `zstd` is both a registered decoder and
    included in the default `Accept-Encoding` header.
    """
    from httpx import _compat, _decoders

    assert _compat.zstandard is not None
    assert "zstd" in _decoders.SUPPORTED_DECODERS
    assert _decoders.SUPPORTED_DECODERS["zstd"] is _decoders.ZStandardDecoder
    assert httpx._client.ACCEPT_ENCODING == "gzip, deflate, br, zstd"

    # Instantiating the decoder without the package raises an ImportError
    # pointing users at the optional extra, rather than failing at import time.
    monkeypatch.setattr(_decoders, "zstandard", None)
    with pytest.raises(ImportError, match="httpx\\[zstd\\]"):
        _decoders.ZStandardDecoder()


def test_zstd_missing_optional_dependency() -> None:
    """
    When the `zstandard` package isn't installed, importing httpx must still
    succeed, `zstd` must not be registered as a decoder, and the default
    Accept-Encoding header must not advertise support for it.

    We run this in an isolated subprocess, with `zstandard` blocked from the
    import machinery.
    """
    code = """
import builtins
import sys

real_import = builtins.__import__


def blocked_import(name, *args, **kwargs):
    if name == "zstandard" or name.startswith("zstandard."):
        raise ImportError(f"No module named {name!r}")
    return real_import(name, *args, **kwargs)


builtins.__import__ = blocked_import

sys.path.insert(0, %r)
import httpx
from httpx._compat import zstandard
from httpx._decoders import SUPPORTED_DECODERS

assert zstandard is None
assert "zstd" not in SUPPORTED_DECODERS
assert httpx._client.ACCEPT_ENCODING == "gzip, deflate, br"
assert (
    httpx.Client().headers["Accept-Encoding"] == "gzip, deflate, br"
)
""" % str(pathlib.Path(__file__).parent.parent)

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_invalid_content_encoding_header():
    headers = [(b"Content-Encoding", b"invalid-header")]
    body = b"test 123"

    response = httpx.Response(
        200,
        headers=headers,
        content=body,
    )
    assert response.content == body
