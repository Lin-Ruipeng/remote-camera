import numpy as np

from remote_camera.capture import FramePacket
from remote_camera.pose import PoseResult
from remote_camera.render import annotate


def test_empty_pose_preserves_frame_and_input_is_not_modified():
    image = np.full((40, 50, 3), 23, dtype=np.uint8)
    result = PoseResult(FramePacket(image, 4, 0), np.empty((0, 5)), np.empty((0, 17, 3)), 1, 1)
    output = annotate(result)
    np.testing.assert_array_equal(output, image)
    assert output is not image


def test_visible_keypoints_align_with_box_on_the_matching_frame():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    points = np.zeros((1, 17, 3), dtype=np.float32)
    points[0, 5] = (30, 40, 0.9)
    points[0, 6] = (70, 40, 0.9)
    # A low confidence point should not be drawn.
    points[0, 9] = (50, 70, 0.1)
    result = PoseResult(FramePacket(image, 4, 0), np.array([[10, 10, 90, 90, 0.9]]), points, 1, 1)
    output = annotate(result)
    assert output[40, 50].any()
    assert output[40, 30].any()
    assert not output[70, 50].any()
    assert not image.any()
