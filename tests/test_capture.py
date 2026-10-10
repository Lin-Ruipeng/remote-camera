import threading
import time

import numpy as np
import pytest

from remote_camera.capture import LatestValue, transform_frame


def test_slow_consumer_gets_latest_frame_without_backlog():
    mailbox = LatestValue()
    mailbox.publish(1)
    sequence, value = mailbox.get()
    assert value == 1
    for frame in range(2, 101):
        mailbox.publish(frame)
    sequence, value = mailbox.get(sequence)
    assert (sequence, value) == (100, 100)
    assert mailbox.get(sequence) == (100, None)


def test_stop_wakes_waiting_consumer():
    mailbox = LatestValue()
    received = []
    consumer = threading.Thread(target=lambda: received.append(mailbox.get(timeout=30)))
    consumer.start()
    mailbox.close()
    consumer.join(timeout=1)
    assert not consumer.is_alive()
    assert received == [(0, None)]


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("mirror", [False, True])
def test_rotation_and_mirror_keep_pixel_coordinates_consistent(rotation, mirror):
    pixels = np.arange(6, dtype=np.uint8).reshape(2, 3)
    image = np.repeat(pixels[:, :, None], 3, axis=2)
    expected = np.rot90(image, -(rotation // 90))
    if mirror:
        expected = expected[:, ::-1]
    transformed = transform_frame(image, rotation, mirror)
    np.testing.assert_array_equal(transformed, expected)
    assert transformed.flags.c_contiguous


def test_invalid_rotation_is_reported():
    with pytest.raises(ValueError):
        transform_frame(np.zeros((2, 3, 3), np.uint8), 45, False)


def test_timeout_does_not_reemit_previous_frame():
    mailbox = LatestValue()
    mailbox.publish("frame")
    sequence, _ = mailbox.get()
    started = time.monotonic()
    assert mailbox.get(sequence, timeout=0.02) == (sequence, None)
    assert time.monotonic() - started >= 0.015
