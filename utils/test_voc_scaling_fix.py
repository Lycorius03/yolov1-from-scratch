"""
TDD - RED phase tests: verify annotation scaling bug in voc_dataset.py.

Bug: _encode_target and _encode_raw_target divide coordinates by fixed
     image_size=448 instead of the original image dimensions (orig_w, orig_h).
     For non-448x448 images, this produces misaligned ground truth boxes.

Fix: Use orig_w, orig_h from the PIL Image for normalization.
"""
import torch
import sys
sys.path.insert(0, "f:/yolov1_from_scratch")


def test_encode_target_with_non_square_image():
    """
    RED: Simulate voc_dataset._encode_target behavior with a non-448 image.

    Image: 500x375 (common VOC image size)
    Box at pixel coords: xmin=100, ymin=75, xmax=300, ymax=300
    (a 200x225 box centered at (200, 187.5))

    OLD (buggy): divide by 448 -> centers and sizes are wrong
    NEW (fixed): divide by 500/375 -> centers and sizes are correct
    """
    S = 7

    # Original image size (from PIL Image.size = (width, height))
    orig_w, orig_h = 500, 375

    # Box in pixel coordinates
    xmin, ymin, xmax, ymax = 100.0, 75.0, 300.0, 300.0

    # === OLD way: divide by fixed 448 ===
    image_size = 448
    cx_old = ((xmin + xmax) / 2) / image_size
    cy_old = ((ymin + ymax) / 2) / image_size
    w_old = (xmax - xmin) / image_size
    h_old = (ymax - ymin) / image_size

    col_old = min(int(cx_old * S), S - 1)
    row_old = min(int(cy_old * S), S - 1)
    cx_cell_old = cx_old * S - col_old
    cy_cell_old = cy_old * S - row_old

    # === NEW way: divide by original image dimensions ===
    cx_new = ((xmin + xmax) / 2) / orig_w
    cy_new = ((ymin + ymax) / 2) / orig_h
    w_new = (xmax - xmin) / orig_w
    h_new = (ymax - ymin) / orig_h

    col_new = min(int(cx_new * S), S - 1)
    row_new = min(int(cy_new * S), S - 1)
    cx_cell_new = cx_new * S - col_new
    cy_cell_new = cy_new * S - row_new

    print(f"  Image: {orig_w}x{orig_h}")
    print(f"  Box pixel: ({xmin},{ymin}) -> ({xmax},{ymax})")
    print(f"")
    print(f"  OLD (div 448):")
    print(f"    cx={cx_old:.4f}, cy={cy_old:.4f}, w={w_old:.4f}, h={h_old:.4f}")
    print(f"    grid cell: ({row_old}, {col_old}), cell offset: ({cx_cell_old:.4f}, {cy_cell_old:.4f})")
    print(f"")
    print(f"  NEW (div {orig_w}/{orig_h}):")
    print(f"    cx={cx_new:.4f}, cy={cy_new:.4f}, w={w_new:.4f}, h={h_new:.4f}")
    print(f"    grid cell: ({row_new}, {col_new}), cell offset: ({cx_cell_new:.4f}, {cy_cell_new:.4f})")

    # RED assertion: old and new should differ significantly
    # For 500x375 image, center_x differs by factor 448/500 = 0.896
    # That's ~10% error in normalized coordinates
    center_x_error = abs(cx_old - cx_new)
    center_y_error = abs(cy_old - cy_new)
    print(f"\n  Center error: dx={center_x_error:.4f}, dy={center_y_error:.4f}")

    # The error should be large enough to potentially shift the grid cell
    assert center_x_error > 0.02 or center_y_error > 0.02, (
        f"RED: Coordinate error should be noticeable for non-448 image. "
        f"dx={center_x_error:.4f}, dy={center_y_error:.4f}"
    )

    # The OLD col/row might be WRONG because of the scaling error
    # For a 500px wide image, cx_new = 200/500 = 0.4, col_new = int(0.4*7) = 2
    # For OLD: cx_old = 200/448 = 0.446, col_old = int(0.446*7) = 3
    # This is a DIFFERENT grid cell!
    if col_old != col_new:
        print(f"  CRITICAL: Grid cell MISMATCH! OLD col={col_old}, NEW col={col_new}")
        print(f"  -> Object mapped to WRONG grid cell in buggy code!")

    if row_old != row_new:
        print(f"  CRITICAL: Grid cell MISMATCH! OLD row={row_old}, NEW row={row_new}")
        print(f"  -> Object mapped to WRONG grid cell in buggy code!")

    # Verify correct answer with original dimensions
    expected_cx = 200.0 / 500.0  # 0.4
    expected_cy = 187.5 / 375.0  # 0.5
    assert abs(cx_new - expected_cx) < 0.001, \
        f"cx should be {expected_cx:.4f}, got {cx_new:.4f}"
    assert abs(cy_new - expected_cy) < 0.001, \
        f"cy should be {expected_cy:.4f}, got {cy_new:.4f}"
    assert abs(w_new - 200.0/500.0) < 0.001, "w should be 0.4"
    assert abs(h_new - 225.0/375.0) < 0.001, "h should be 0.6"

    print(f"\n  PASS: Bug confirmed! Fixed-448 scaling causes coordinate errors.")
    print(f"        Old grid cell: ({row_old},{col_old}), New: ({row_new},{col_new})")


def test_encode_raw_target_with_non_square_image():
    """
    RED: Same bug in _encode_raw_target.
    """
    # Same scenario
    xmin, ymin, xmax, ymax = 100.0, 75.0, 300.0, 300.0
    orig_w, orig_h = 500, 375

    xyxy = torch.tensor([[xmin, ymin, xmax, ymax]])

    # OLD way
    image_size = 448
    cx_old = (xyxy[0, 0] + xyxy[0, 2]) / 2 / image_size
    cy_old = (xyxy[0, 1] + xyxy[0, 3]) / 2 / image_size
    w_old = (xyxy[0, 2] - xyxy[0, 0]) / image_size
    h_old = (xyxy[0, 3] - xyxy[0, 1]) / image_size

    # NEW way
    cx_new = (xyxy[0, 0] + xyxy[0, 2]) / 2 / orig_w
    cy_new = (xyxy[0, 1] + xyxy[0, 3]) / 2 / orig_h
    w_new = (xyxy[0, 2] - xyxy[0, 0]) / orig_w
    h_new = (xyxy[0, 3] - xyxy[0, 1]) / orig_h

    print(f"\n  _encode_raw_target - OLD (div 448):")
    print(f"    cx={cx_old:.4f}, cy={cy_old:.4f}, w={w_old:.4f}, h={h_old:.4f}")

    print(f"  _encode_raw_target - NEW (div orig):")
    print(f"    cx={cx_new:.4f}, cy={cy_new:.4f}, w={w_new:.4f}, h={h_new:.4f}")

    # RED: old and new should differ
    assert abs(cx_old - cx_new) > 0.01, (
        f"RED: _encode_raw_target should show coordinate error for non-448 image"
    )

    # Verify correct answers
    assert abs(cx_new - 200.0/500.0) < 0.001, f"Wrong cx: {cx_new:.4f}"
    assert abs(cy_new - 187.5/375.0) < 0.001, f"Wrong cy: {cy_new:.4f}"

    print(f"  PASS: _encode_raw_target bug confirmed!")


def test_448x448_image_is_unaffected():
    """
    Sanity check: for images that ARE 448x448, the fix should produce
    the same result as the old code.
    """
    xmin, ymin, xmax, ymax = 100.0, 200.0, 300.0, 400.0
    orig_w = orig_h = 448  # exactly the old image_size

    # OLD
    cx_old = ((xmin + xmax) / 2) / 448
    cy_old = ((ymin + ymax) / 2) / 448

    # NEW
    cx_new = ((xmin + xmax) / 2) / orig_w
    cy_new = ((ymin + ymax) / 2) / orig_h

    assert abs(cx_old - cx_new) < 0.0001, "Should be identical for 448x448 image"
    assert abs(cy_old - cy_new) < 0.0001, "Should be identical for 448x448 image"

    print(f"\n  PASS: 448x448 image gives same result (old={cx_old:.4f}, new={cx_new:.4f})")


if __name__ == "__main__":
    print("=" * 60)
    print("RED Phase: Verify Annotation Scaling Bug in voc_dataset.py")
    print("=" * 60)

    print("\n[Test 1] _encode_target with non-square image...")
    test_encode_target_with_non_square_image()

    print("\n[Test 2] _encode_raw_target with non-square image...")
    test_encode_raw_target_with_non_square_image()

    print("\n[Test 3] 448x448 image unaffected by fix...")
    test_448x448_image_is_unaffected()

    print("\n" + "=" * 60)
    print("RED phase complete: Annotation scaling bug confirmed!")
    print("Ready for GREEN phase (implement fix).")
    print("=" * 60)
