import cv2
import numpy as np
CLOAK_COLOR = "red"
KERNEL = np.ones((3, 3), np.uint8)
def get_color_ranges(color: str):
    if color.lower() == "red":
        ranges = [
            (np.array([0, 120, 70]),   np.array([10, 255, 255])),
            (np.array([170, 120, 70]), np.array([180, 255, 255])),
        ]
    elif color.lower() == "blue":
        ranges = [(np.array([94, 80, 2]), np.array([126, 255, 255]))]
    elif color.lower() == "green":
        ranges = [(np.array([35, 40, 40]), np.array([85, 255, 255]))]
    else:
        raise ValueError("CLOAK_COLOR must be 'red', 'blue', or 'green'")
    return ranges
def build_mask(hsv, ranges):
    masks = []
    for low, high in ranges:
        masks.append(cv2.inRange(hsv, low, high))
    mask = masks[0]
    for m in masks[1:]:
        mask = mask | m
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=2)
    mask = cv2.dilate(mask, KERNEL, iterations=1)
    return mask
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return
    print("🪄 Invisibility Cloak")
    print("Instructions:")
    print("  • Put the cloak out of frame and press 'b' to capture background.")
    print("  • Then bring the cloak into frame to see the effect.")
    print("  • Press 's' to save a snapshot, 'q' to quit.\n")
    bg = None
    ranges = get_color_ranges(CLOAK_COLOR)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame grab failed; exiting.")
            break
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        if bg is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = build_mask(hsv, ranges)
            mask_inv = cv2.bitwise_not(mask)
            cloak_bg = cv2.bitwise_and(bg, bg, mask=mask)       
            rest = cv2.bitwise_and(frame, frame, mask=mask_inv)     
            output = cv2.addWeighted(cloak_bg, 1.0, rest, 1.0, 0.0)
            display = output
        else:
            cv2.putText(display, "Press 'b' to capture background",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, "Press 'b' to capture background",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Invisibility Cloak", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('b'):
            print("📸 Capturing background...")
            frames = []
            for _ in range(40):
                ok, f = cap.read()
                if not ok:
                    continue
                f = cv2.flip(f, 1)
                frames.append(f.astype(np.float32))
                cv2.waitKey(10)
            if frames:
                avg = np.mean(frames, axis=0).astype(np.uint8)
                bg = avg
                print("✅ Background set.")
            else:
                print("⚠️ Could not capture background, try again.")
        elif key == ord('s'):
            filename = "cloak_snapshot.png"
            cv2.imwrite(filename, display)
            print(f"💾 Saved snapshot to {filename}")
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()