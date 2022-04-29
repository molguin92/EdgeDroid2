#  Copyright (c) 2022 Manuel Olguín Muñoz <molguin@kth.se>
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

import socket
import struct
from typing import Generator, NamedTuple, Tuple

import numpy as np
import numpy.typing as npt
from loguru import logger

HEADER_PACK_FMT = "!5I"  # [seq][height][width][channels][data length]
HEADER_LEN = struct.calcsize(HEADER_PACK_FMT)
IMG_BYTES_ORDER = "C"

RESP_PACK_FMT = "!?5I"
# [success][height][width][channels][img data length][text data length]
RESP_LEN = struct.calcsize(RESP_PACK_FMT)


class Frame(NamedTuple):
    seq: int
    image_data: npt.NDArray


class Response(NamedTuple):
    transition: bool
    image_guidance: npt.NDArray
    text_guidance: str


def bytes_to_numpy_image(
    data: bytes, height: int, width: int, channels: int
) -> npt.NDArray:
    return np.frombuffer(data, dtype=np.uint8).reshape(
        (height, width, channels), order=IMG_BYTES_ORDER
    )


def recv_from_socket(sock: socket.SocketType, amount: int) -> bytes:
    received = b""
    while len(received) < amount:
        data = sock.recv(amount - len(received))
        if len(data) == 0:
            # socket is closed
            raise EOFError()
        else:
            received += data
    return received


def pack_frame(seq: int, image_frame: npt.NDArray) -> bytes:
    height, width, channels = image_frame.shape
    img_data = image_frame.tobytes(order=IMG_BYTES_ORDER)
    data_len = len(img_data)
    hdr = struct.pack(HEADER_PACK_FMT, seq, height, width, channels, data_len)
    return hdr + img_data


def frame_stream_unpack(
    sock: socket.SocketType,
) -> Generator[Frame, None, None]:
    # TODO: does this need to be optimized somehow?
    logger.info("Started frame stream unpacker")
    try:
        while True:
            header = recv_from_socket(sock, HEADER_LEN)
            # got a complete header
            seq, height, width, channels, data_len = struct.unpack(
                HEADER_PACK_FMT, header
            )

            # read data
            image_data = recv_from_socket(sock, data_len)

            # got all the data!
            image = bytes_to_numpy_image(image_data, height, width, channels)
            yield Frame(seq, image)
    except EOFError:
        logger.warning("Socket was closed")
    finally:
        logger.debug("Closing frame stream unpacker")


def pack_response(
    transition: bool, image_guidance: npt.NDArray, text_guidance: str
) -> bytes:
    height, width, channels = image_guidance.shape
    img_data = image_guidance.tobytes(order=IMG_BYTES_ORDER)
    img_len = len(img_data)

    text_data = text_guidance.encode("utf8")
    text_len = len(text_data)

    header = struct.pack(
        RESP_PACK_FMT, transition, height, width, channels, img_len, text_len
    )

    return header + img_data + text_data


def response_stream_unpack(
    sock: socket.SocketType,
) -> Generator[Response, None, None]:
    try:
        while True:
            resp_header = recv_from_socket(sock, RESP_LEN)

            # unpack the header into its constituent parts
            transition, height, width, channels, img_len, text_len = struct.unpack(
                RESP_PACK_FMT, resp_header
            )

            img_data = recv_from_socket(sock, img_len)
            text_data = recv_from_socket(sock, text_len)

            guidance_image = bytes_to_numpy_image(img_data, height, width, channels)
            guidance_text = text_data.decode("utf8")

            yield transition, guidance_image, guidance_text
    except EOFError:
        logger.warning("Socket was closed")
    finally:
        logger.debug("Closing response stream unpacker")
