import cv2
import numpy as np
import chess
import chess.engine
import time
import os

# --- PCA9685 & Servo Control Imports ---
try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685
    from adafruit_motor import servo as adafruit_servo_motor
    PCA_AVAILABLE = True
except ImportError:
    print("WARN: Adafruit PCA9685/Motor libraries not found. Arm control will be disabled.")
    PCA_AVAILABLE = False

# --- BoardDetector CLASS (Re-tuned for Recognition and Smoothness) ---
class BoardDetector:
    def __init__(self, output_size=480):
        self.output_size = output_size      # For the final warped board image
        self.last_known_corners = None
        self.consecutive_misses = 0

        # --- Parameters to TUNE for DETECTION, SMOOTHNESS, and ROBUSTNESS ---
        self.processing_width = 480  # Resize input image to this width for consistent processing
        
        self.blur_kernel_size = (5,5)  # Applied to the resized image
        self.canny_threshold1 = 30     # Lower Canny threshold for resized image. **TUNE THIS!**
        self.canny_threshold2 = 100    # Higher Canny threshold for resized image. **TUNE THIS!**
        
        self.approx_poly_epsilon_factor = 0.025 # For contour simplification. Increased slightly.
        self.min_area_ratio = 0.05     # Min contour area relative to *processing_width* image. Was 0.02.
        self.max_area_ratio = 0.95     # Max contour area.

        # Smoothing parameters for detected corners
        self.corner_smoothing_alpha = 0.5 # Was 0.3. Start with moderate smoothing.
        self.max_corner_drift_for_smoothing = 75 # Max avg pixel drift (on resized image) for smoothing. Was 70.

        # ROI (Region of Interest) search parameters
        self.max_consecutive_misses_before_reset = 10
        self.roi_search_expansion_factor = 1.25 # Was 1.35. Slightly smaller expansion if detection is good.
        # --- End of Tunable Parameters ---

    def _order_points(self, pts):
        # Standard method: TL, TR, BR, BL
        # TL = min(x+y), BR = max(x+y)
        # TR = min(y-x), BL = max(y-x)
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-left
        rect[2] = pts[np.argmax(s)] # Bottom-right

        diff = np.diff(pts, axis=1) # Computes element_wise pts[:,1] - pts[:,0]  (y - x)
        rect[1] = pts[np.argmin(diff)] # Top-right
        rect[3] = pts[np.argmax(diff)] # Bottom-left
        
        indices = [np.argmin(s), np.argmax(s), np.argmin(diff), np.argmax(diff)]
        if len(set(indices)) < 4:
             # print("WARN: Degenerate shape in _order_points, returning None.")
             return None 
        return rect

    def _find_board_contour_in_image(self,img_search,img_area_ratios_coeff,is_roi,dbg=False,orig_img_dbg_for_raw_contour_window=None):
        gray=cv2.cvtColor(img_search,cv2.COLOR_BGR2GRAY)
        blr=cv2.GaussianBlur(gray,self.blur_kernel_size,0)

        ct1, ct2 = self.canny_threshold1, self.canny_threshold2
        if is_roi:
            ct1_roi, ct2_roi = int(ct1 * 0.7), int(ct2 * 0.7) 
            ct1, ct2 = max(10, ct1_roi), max(30, ct2_roi)     

        edg=cv2.Canny(blr,ct1,ct2)
        if dbg: 
            if is_roi: cv2.imshow("Debug - ROI Canny Edges", edg)
            else: cv2.imshow("Debug - Full Canny Edges", edg)

        cnts,_=cv2.findContours(edg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:return None

        cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:5]

        for c in cnts:
            peri=cv2.arcLength(c,True)
            approx_poly=cv2.approxPolyDP(c,self.approx_poly_epsilon_factor*peri,True)

            if len(approx_poly)==4:
                contour_area=cv2.contourArea(approx_poly)
                is_convex=cv2.isContourConvex(approx_poly)
                
                min_a = self.min_area_ratio * img_area_ratios_coeff
                max_a = self.max_area_ratio * img_area_ratios_coeff

                if min_a < contour_area < max_a and is_convex:
                    if dbg and not is_roi and orig_img_dbg_for_raw_contour_window is not None:
                        sbi_debug_contour = img_search.copy() 
                        cv2.drawContours(sbi_debug_contour,[approx_poly.reshape(4,2)],-1,(0,255,0),2)
                        cv2.imshow("Debug - 05 Raw Contour (on processed img)",sbi_debug_contour)
                    return approx_poly.reshape(4,2).astype(np.float32)
        return None

    def detect(self,image,debug_mode=False):
        orig_h, orig_w = image.shape[:2]
        
        scale_ratio = self.processing_width / orig_w if orig_w > 0 else 1.0 # Avoid division by zero
        proc_h = int(orig_h * scale_ratio)
        proc_w = self.processing_width
        
        if proc_w <= 0 or proc_h <= 0: # Safety check for resize dimensions
            if debug_mode: print(f"DBG: Invalid processing dimensions {proc_w}x{proc_h}. Skipping detection.")
            return None, None
            
        processed_image = cv2.resize(image, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

        proc_img_area = proc_h * proc_w
        candidate_raw_corners_on_proc_img = None
        roi_was_used=False
        
        last_known_corners_scaled = None
        if self.last_known_corners is not None:
            last_known_corners_scaled = self.last_known_corners * scale_ratio

        if last_known_corners_scaled is not None and \
           self.consecutive_misses < self.max_consecutive_misses_before_reset: 

            x_coords = last_known_corners_scaled[:,0]
            y_coords = last_known_corners_scaled[:,1]
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            min_y, max_y = np.min(y_coords), np.max(y_coords)

            center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
            roi_w = int((max_x - min_x) * self.roi_search_expansion_factor)
            roi_h = int((max_y - min_y) * self.roi_search_expansion_factor)

            roi_x1 = max(0, int(center_x - roi_w / 2))
            roi_y1 = max(0, int(center_y - roi_h / 2))
            roi_x2 = min(proc_w, roi_x1 + roi_w) 
            roi_y2 = min(proc_h, roi_y1 + roi_h) 

            if (roi_x2 - roi_x1) > 20 and (roi_y2 - roi_y1) > 20:
                roi_img = processed_image[roi_y1:roi_y2, roi_x1:roi_x2]
                if roi_img.size > 0:
                    roi_area = roi_img.shape[0] * roi_img.shape[1]
                    corners_in_roi = self._find_board_contour_in_image(roi_img, roi_area, True, debug_mode)
                    if corners_in_roi is not None:
                        candidate_raw_corners_on_proc_img = corners_in_roi + np.array([roi_x1, roi_y1], dtype=np.float32)
                        roi_was_used=True
                        if debug_mode: print("DBG: ROI Hit")
        
        if candidate_raw_corners_on_proc_img is None:
            if debug_mode and last_known_corners_scaled is not None and \
               self.consecutive_misses < self.max_consecutive_misses_before_reset : 
                print("DBG: ROI Miss, Full Search on processed image")
            candidate_raw_corners_on_proc_img = self._find_board_contour_in_image(
                processed_image, proc_img_area, False, debug_mode, 
                orig_img_dbg_for_raw_contour_window=processed_image if debug_mode else None
            )
            roi_was_used = False

        final_corners_on_proc_img = None
        if candidate_raw_corners_on_proc_img is not None:
            ordered_raw_corners = self._order_points(candidate_raw_corners_on_proc_img)
            if ordered_raw_corners is None:
                 if debug_mode: print("DBG: Corner ordering failed.")
            else:
                if last_known_corners_scaled is not None:
                    avg_drift = np.mean(np.linalg.norm(ordered_raw_corners - last_known_corners_scaled, axis=1))
                    if avg_drift < self.max_corner_drift_for_smoothing:
                        final_corners_on_proc_img = (self.corner_smoothing_alpha * ordered_raw_corners + \
                                                     (1 - self.corner_smoothing_alpha) * last_known_corners_scaled)
                        if debug_mode: print(f"DBG: Smoothed (Drift: {avg_drift:.1f}px on proc_img)")
                    else:
                        final_corners_on_proc_img = ordered_raw_corners
                        if debug_mode: print(f"DBG: Raw (Drift {avg_drift:.1f}px > {self.max_corner_drift_for_smoothing}px on proc_img)")
                else:
                    final_corners_on_proc_img = ordered_raw_corners
                    if debug_mode: print("DBG: Raw (No previous corners, on proc_img)")
                
                if final_corners_on_proc_img is not None : # Check before division
                    self.last_known_corners = final_corners_on_proc_img / scale_ratio if scale_ratio != 0 else final_corners_on_proc_img
                self.consecutive_misses = 0
        else:
            self.consecutive_misses += 1
            if last_known_corners_scaled is not None and \
               self.consecutive_misses < self.max_consecutive_misses_before_reset:
                final_corners_on_proc_img = last_known_corners_scaled 
                if debug_mode: print(f"DBG: Coasting (Miss #{self.consecutive_misses}) on proc_img")
            else:
                if debug_mode and self.last_known_corners is not None: print("DBG: Max misses, resetting last_known_corners.")
                self.last_known_corners = None
                final_corners_on_proc_img = None

        if final_corners_on_proc_img is None:
            return None, None

        final_corners_original_scale = final_corners_on_proc_img / scale_ratio if scale_ratio != 0 else final_corners_on_proc_img


        destination_corners_for_warp = np.array([
            [0,0], [self.output_size-1, 0],
            [self.output_size-1, self.output_size-1], [0, self.output_size-1]], dtype="float32")

        try:
            transform_matrix = cv2.getPerspectiveTransform(final_corners_original_scale, destination_corners_for_warp)
            warped_board_img = cv2.warpPerspective(image, transform_matrix, (self.output_size, self.output_size))
        except cv2.error as e:
            if debug_mode: print(f"DBG: Warp Perspective Error: {e}, Corners (orig scale): {final_corners_original_scale}")
            return None, final_corners_original_scale

        if debug_mode:
            debug_display_img = image.copy()
            if roi_was_used and 'roi_x1' in locals() and scale_ratio !=0: 
                 roi_x1_orig, roi_y1_orig = int(roi_x1 / scale_ratio), int(roi_y1 / scale_ratio)
                 roi_x2_orig, roi_y2_orig = int(roi_x2 / scale_ratio), int(roi_y2 / scale_ratio)
                 cv2.rectangle(debug_display_img,(roi_x1_orig,roi_y1_orig),(roi_x2_orig,roi_y2_orig),(255,165,0),2)

            corner_colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)] 
            for i, p_corner in enumerate(final_corners_original_scale): 
                cv2.circle(debug_display_img, tuple(np.int0(p_corner)), 7, corner_colors[i], -1)
                cv2.putText(debug_display_img, str(i), tuple(np.int0(p_corner)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            cv2.imshow("Debug - 06 Active Corners (on original img)", debug_display_img)

        return warped_board_img, final_corners_original_scale

# --- Arm Controller Class ---
class ArmController:
    SERVO_CHANNELS = {
        'base': 0, 'shoulder': 1, 'elbow': 2,
        'wrist_ud': 3, 'wrist_rot': 4, 'claw': 5,
    }
    SERVO_PARAMS = { 
        'base':      {'min_pulse': 500, 'max_pulse': 2500, 'default_angle': 90,  'min_angle': 0,   'max_angle': 180, 'current_angle': 90},
        'shoulder':  {'min_pulse': 500, 'max_pulse': 2500, 'default_angle': 135, 'min_angle': 30,  'max_angle': 150, 'current_angle': 135},
        'elbow':     {'min_pulse': 500, 'max_pulse': 2500, 'default_angle': 45,  'min_angle': 0,   'max_angle': 135, 'current_angle': 45},
        'wrist_ud':  {'min_pulse': 500, 'max_pulse': 2500, 'default_angle': 90,  'min_angle': 0,   'max_angle': 180, 'current_angle': 90},
        'wrist_rot': {'min_pulse': 500, 'max_pulse': 2500, 'default_angle': 90,  'min_angle': 0,   'max_angle': 180, 'current_angle': 90},
        'claw':      {'min_pulse': 600, 'max_pulse': 2400, 'default_angle': 45,  'min_angle': 45,  'max_angle': 110, 'current_angle': 45}
    }
    def __init__(self):
        self.servos = {}
        self.pca = None
        if not PCA_AVAILABLE:
            print("ArmController: PCA9685 libraries not available. Arm disabled.")
            return
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = PCA9685(i2c)
            self.pca.frequency = 50
            print("PCA9685 initialized successfully.")
            for name, channel_num in self.SERVO_CHANNELS.items():
                if name not in self.SERVO_PARAMS:
                    print(f"Warning: No params found for servo '{name}', skipping initialization.")
                    continue
                params = self.SERVO_PARAMS[name]
                actuation_range_for_lib = 180
                servo_obj = adafruit_servo_motor.Servo(
                    self.pca.channels[channel_num], min_pulse=params['min_pulse'],
                    max_pulse=params['max_pulse'], actuation_range=actuation_range_for_lib)
                self.servos[name] = servo_obj
                params['current_angle'] = params['default_angle']
                self.set_servo_angle(name, params['default_angle'], initial_setup=True)
                time.sleep(0.05)
            print("Servos initialized (attempted). VERIFY POSITIONS AND CALIBRATE!")
        except ValueError as e:
            print(f"CRITICAL Error: Could not find PCA9685. Check I2C. Error: {e}")
            self.pca = None
        except Exception as e:
            print(f"CRITICAL Error during ArmController init: {e}")
            self.pca = None

    def set_servo_angle(self, servo_name, angle, initial_setup=False):
        if not self.pca or servo_name not in self.servos: return
        servo_obj, params = self.servos[servo_name], self.SERVO_PARAMS[servo_name]
        clamped_angle = max(params['min_angle'], min(params['max_angle'], float(angle)))
        try:
            servo_obj.angle = clamped_angle
            params['current_angle'] = clamped_angle
        except Exception as e: print(f"Error setting servo '{servo_name}': {e}")

    def go_to_home_position(self):
        if not self.pca: print("Arm not ready (PCA)."); return
        print("Arm moving to home position...")
        order = ['elbow', 'shoulder', 'wrist_ud', 'wrist_rot', 'base', 'claw']
        for name in order:
            if name in self.SERVO_PARAMS:
                self.set_servo_angle(name, self.SERVO_PARAMS[name]['default_angle']); time.sleep(0.2)
        print("Arm at home position.")

    def operate_gripper(self, state, position=None):
        if not self.pca: print("Arm not ready (PCA)."); return
        params = self.SERVO_PARAMS['claw']
        target_angle = position if position is not None else (params['min_angle'] if state == "open" else (params['max_angle'] if state == "close" else None))
        if target_angle is None: print(f"Gripper: Unknown state '{state}'."); return
        print(f"Gripper: {state.capitalize()}")
        self.set_servo_angle('claw', target_angle); time.sleep(0.3)

    def test_servo_increment(self, servo_name, increment=5):
        if not self.pca or servo_name not in self.SERVO_PARAMS: return
        current = self.SERVO_PARAMS[servo_name]['current_angle']
        new_angle = current + increment
        print(f"Servo '{servo_name}' Test: current {current:.1f}, target {new_angle:.1f}")
        self.set_servo_angle(servo_name, new_angle)

    def test_servo_decrement(self, servo_name, decrement=5):
        if not self.pca or servo_name not in self.SERVO_PARAMS: return
        current = self.SERVO_PARAMS[servo_name]['current_angle']
        new_angle = current - decrement
        print(f"Servo '{servo_name}' Test: current {current:.1f}, target {new_angle:.1f}")
        self.set_servo_angle(servo_name, new_angle) # Corrected line

    def print_servo_angles(self):
        if not self.pca: print("Arm not ready (PCA)."); return
        print("\n--- Current Servo Angles ---")
        for name, params in self.SERVO_PARAMS.items():
            print(f"Servo '{name}': {params['current_angle']:.1f}° (Min: {params['min_angle']}, Max: {params['max_angle']}, Default: {params['default_angle']})")
        print("---------------------------\nUpdate SERVO_PARAMS in script to save these.")

    def move_piece_on_board(self, from_sq_uci, to_sq_uci, captured_piece=False):
        if not self.pca: print("Arm not ready (PCA)."); return
        print(f"ARM (Placeholder): Move {from_sq_uci} to {to_sq_uci}"); time.sleep(1)

    def release_servos(self):
        if not self.pca: print("PCA not init."); return
        print("Releasing servos...")
        for i in range(16):
            try: self.pca.channels[i].duty_cycle = 0
            except: pass
        print("Servos released.")


# --- Initialize Webcam ---
cap = None; CAM_INDEX = -1
MAX_CAM_INDICES_TO_TRY = 4
for cam_idx_loop in range(MAX_CAM_INDICES_TO_TRY):
    cap_test = cv2.VideoCapture(cam_idx_loop)
    if cap_test.isOpened():
        cap, CAM_INDEX = cap_test, cam_idx_loop
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Webcam OK (index {CAM_INDEX}, W: {w}, H: {h}).")
        break
    else: cap_test.release()
if cap is None: print("CRITICAL: Webcam init failed.")


# --- Load Stockfish Engine ---
engine = None; engine_path = "/usr/games/stockfish"
try:
    if not (os.path.exists(engine_path) and os.access(engine_path, os.X_OK)):
        raise FileNotFoundError("Stockfish not found/executable. Try: sudo apt install stockfish")
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    print(f"Stockfish engine OK: {engine_path}")
except Exception as e: print(f"Error loading Stockfish: {e}")


# --- Chess Game and Utilities ---
current_chess_board = chess.Board()

# --- Main Loop ---
def main():
    global current_chess_board, cap, CAM_INDEX, engine 

    if cap is None: print("Cannot start: Webcam not available."); return

    board_detector = BoardDetector(output_size=480)
    arm_controller = ArmController() 

    RUN_IN_DEBUG_MODE = False 
    print_keys_help()

    active_test_servo = 'base'; engine_play_requested = False
    if arm_controller.pca is None: print("INFO: ARM CONTROLLER DISABLED (PCA init failed or libs missing).")

    LOG_PERFORMANCE = False; loop_time_filter = []
    start_time_perf = 0 # Initialize for linter

    while True:
        if LOG_PERFORMANCE: start_time_perf = time.monotonic()
        
        should_break_loop = False # Flag to break from outer loop if camera fails critically
        ret, frame = cap.read()
        if not ret or frame is None: 
            if handle_camera_failure(): # handle_camera_failure returns True if critical and should break
                should_break_loop = True
            if should_break_loop: break
            continue


        processed_frame_for_detector = frame.copy()
        warped_board_img, board_corners = board_detector.detect(processed_frame_for_detector, debug_mode=RUN_IN_DEBUG_MODE)
        
        display_frame = frame.copy()
        if warped_board_img is not None:
            cv2.imshow("1. Warped Board", warped_board_img)
            if board_corners is not None: cv2.polylines(display_frame, [np.int32(board_corners)], True, (0,255,0), 2, cv2.LINE_AA)
        else: display_placeholder_board(board_detector.output_size)
        
        update_display_overlays(display_frame, active_test_servo, LOG_PERFORMANCE, loop_time_filter, start_time_perf if LOG_PERFORMANCE else 0)
        cv2.imshow("3. Live Feed with Detections", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        RUN_IN_DEBUG_MODE, engine_play_requested, active_test_servo, should_quit_flag = \
            handle_key_presses(key, RUN_IN_DEBUG_MODE, engine_play_requested, active_test_servo, 
                               current_chess_board, board_detector, arm_controller)
        if should_quit_flag: break

        if engine_play_requested and arm_controller.pca and engine and not current_chess_board.is_game_over():
            engine_play_requested = False; print("Engine thinking..."); time.sleep(0.1)
            try:
                result = engine.play(current_chess_board, chess.engine.Limit(time=0.3, depth=6))
                if result.move:
                    print(f"Engine plays: {result.move.uci()}")
                    arm_controller.move_piece_on_board(result.move.uci()[:2], result.move.uci()[2:], current_chess_board.is_capture(result.move))
                    current_chess_board.push(result.move); print(current_chess_board)
                else: print("Engine found no move.")
            except Exception as e: print(f"Engine/Arm play error: {e}")
        elif engine_play_requested: 
            engine_play_requested = False
            if not arm_controller.pca: print("Engine move ignored: Arm not ready.")
            elif not engine: print("Engine move ignored: Engine not ready.")
            elif current_chess_board.is_game_over(): print("Engine move ignored: Game over.")


        if current_chess_board.is_game_over() and not RUN_IN_DEBUG_MODE:
            print("Game Over.")

    cleanup_resources(arm_controller, engine, cap)
    print_game_result(current_chess_board)

def print_keys_help():
    print("\n--- Chess Vision Arm Control ---")
    print("  'q':Quit | 'd':Toggle BoardDetect Debug")
    print("  Game: 'm':Engine Move | 'u':Undo | 'r':Reset")
    print("  Arm:  'h':Home | 'o':Open | 'c':Close | 'l':Release | 'p':Print Angles")
    print("        '1-6':Select Servo | '+/-':Increment/Decrement")

def handle_camera_failure():
    global cap, CAM_INDEX 
    print("ERR: Frame skip/empty. Retrying camera..."); time.sleep(0.1)
    if cap is not None: cap.release()
    cap = None # Explicitly set cap to None after release
    time.sleep(0.5)
    if CAM_INDEX != -1:
        temp_cap = cv2.VideoCapture(CAM_INDEX)
        if temp_cap.isOpened():
            cap = temp_cap; print("INFO: Re-opened camera.")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return False # Camera re-opened, don't quit
        else:
            print(f"CRITICAL: Failed to re-open @{CAM_INDEX}."); temp_cap.release()
            return True # Critical failure, signal quit
    else: 
        print("CRITICAL: Cam index unknown."); 
        return True # Critical failure, signal quit

def display_placeholder_board(output_size):
    dummy = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    cv2.putText(dummy, "No Board", (20, output_size // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.imshow("1. Warped Board", dummy)

def update_display_overlays(display_frame, active_test_servo, log_perf, time_filter, start_t):
    cv2.putText(display_frame, f"TestServo:{active_test_servo.upper()}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1, cv2.LINE_AA)
    cv2.putText(display_frame, "Keys:m,u,r,q,d|Arm:h,o,c,l,1-6,p,+-", (10,display_frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,50),1,cv2.LINE_AA)
    if log_perf and start_t > 0 : # Check start_t also
        proc_time = time.monotonic() - start_t
        time_filter.append(proc_time)
        if len(time_filter) > 20: time_filter.pop(0)
        avg_fps = 1.0 / (sum(time_filter) / len(time_filter)) if time_filter else 0
        cv2.putText(display_frame, f"FPS:{avg_fps:.1f}", (display_frame.shape[1]-100,25), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1,cv2.LINE_AA)

def handle_key_presses(key, run_debug, eng_req, act_servo, board_state, board_det, arm_ctrl):
    should_quit_flag = False
    can_arm = arm_ctrl.pca is not None

    if key == ord('q'): should_quit_flag = True; print("Quitting.")
    elif key == ord('d'):
        run_debug = not run_debug
        print(f"BoardDetector Debug Mode: {'ON' if run_debug else 'OFF'}")
        if not run_debug: 
            windows_to_try_close = ["Debug - ROI Canny Edges", "Debug - Full Canny Edges", 
                                   "Debug - 05 Raw Contour (on processed img)", 
                                   "Debug - 06 Active Corners (on original img)"]
            for win_name in windows_to_try_close:
                try: cv2.destroyWindow(win_name)
                except cv2.error: pass 
    elif key == ord('m'):
        if can_arm and engine: eng_req = True; print("Engine move requested.")
        elif not engine: print("Engine not ready for 'm'.")
        else: print("Arm not ready for 'm'.")
    elif key == ord('u'):
        if board_state.move_stack: board_state.pop()
        if board_state.move_stack: board_state.pop()
        print("Undo."); print(board_state)
    elif key == ord('r'):
        board_state.reset(); board_det.last_known_corners=None; board_det.consecutive_misses=0
        if can_arm: arm_ctrl.go_to_home_position()
        print("Board & Arm Reset."); print(board_state)
    elif key == ord('h') and can_arm: arm_ctrl.go_to_home_position()
    elif key == ord('o') and can_arm: arm_ctrl.operate_gripper("open")
    elif key == ord('c') and can_arm: arm_ctrl.operate_gripper("close")
    elif key == ord('l') and can_arm: arm_ctrl.release_servos()
    elif key == ord('p') and can_arm: arm_ctrl.print_servo_angles()
    elif ord('1') <= key <= ord('6'):
        servo_map = {ord('1'):'base', ord('2'):'shoulder', ord('3'):'elbow', ord('4'):'wrist_ud', ord('5'):'wrist_rot', ord('6'):'claw'}
        if key in servo_map: act_servo = servo_map[key]; print(f"Selected Servo: {act_servo.upper()}")
    elif (key == ord('+') or key == ord('=')) and can_arm: arm_ctrl.test_servo_increment(act_servo, 5)
    elif key == ord('-') and can_arm: arm_ctrl.test_servo_decrement(act_servo, 5)
    
    return run_debug, eng_req, act_servo, should_quit_flag

def cleanup_resources(arm_controller, stockfish_engine, camera_capture):
    print("Cleaning up...")
    if arm_controller.pca is not None: arm_controller.release_servos()
    if stockfish_engine:
        try: stockfish_engine.quit()
        except: pass
    if camera_capture: camera_capture.release()
    cv2.destroyAllWindows()

def print_game_result(board_state):
    if board_state.is_checkmate(): print(f"Checkmate! {'White' if not board_state.turn else 'Black'} wins.")
    elif board_state.is_stalemate(): print("Stalemate! Draw.")
    elif board_state.is_insufficient_material(): print("Draw by insufficient material.")
    elif board_state.is_seventyfive_moves(): print("Draw by 75-move rule.")
    elif board_state.is_fivefold_repetition(): print("Draw by fivefold repetition.")


if __name__ == "__main__":
    main()