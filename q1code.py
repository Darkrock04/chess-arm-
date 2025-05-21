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

# --- BoardDetector CLASS (From previous version) ---
class BoardDetector:
    def __init__(self, output_size=480):
        self.output_size = output_size
        self.last_known_corners = None
        self.consecutive_misses = 0
        self.processing_width = 480
        self.blur_kernel_size = (5,5)
        self.canny_threshold1 = 30
        self.canny_threshold2 = 100
        self.approx_poly_epsilon_factor = 0.025
        self.min_area_ratio = 0.05
        self.max_area_ratio = 0.95
        self.corner_smoothing_alpha = 0.5
        self.max_corner_drift_for_smoothing = 75
        self.max_consecutive_misses_before_reset = 10
        self.roi_search_expansion_factor = 1.25

    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        indices = [np.argmin(s), np.argmax(s), np.argmin(diff), np.argmax(diff)]
        if len(set(indices)) < 4: return None
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
        scale_ratio = self.processing_width / orig_w if orig_w > 0 else 1.0
        proc_h = int(orig_h * scale_ratio)
        proc_w = self.processing_width
        if proc_w <= 0 or proc_h <= 0:
            if debug_mode: print(f"DBG: Invalid processing dimensions {proc_w}x{proc_h}.")
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
            x_coords = last_known_corners_scaled[:,0]; y_coords = last_known_corners_scaled[:,1]
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
            roi_w = int((max_x - min_x) * self.roi_search_expansion_factor)
            roi_h = int((max_y - min_y) * self.roi_search_expansion_factor)
            roi_x1 = max(0, int(center_x - roi_w / 2)); roi_y1 = max(0, int(center_y - roi_h / 2))
            roi_x2 = min(proc_w, roi_x1 + roi_w); roi_y2 = min(proc_h, roi_y1 + roi_h)
            if (roi_x2 - roi_x1) > 20 and (roi_y2 - roi_y1) > 20:
                roi_img = processed_image[roi_y1:roi_y2, roi_x1:roi_x2]
                if roi_img.size > 0:
                    corners_in_roi = self._find_board_contour_in_image(roi_img, roi_img.shape[0]*roi_img.shape[1], True, debug_mode)
                    if corners_in_roi is not None:
                        candidate_raw_corners_on_proc_img = corners_in_roi + np.array([roi_x1, roi_y1], dtype=np.float32)
                        roi_was_used=True
                        if debug_mode: print("DBG: ROI Hit")
        if candidate_raw_corners_on_proc_img is None:
            if debug_mode and last_known_corners_scaled is not None and self.consecutive_misses < self.max_consecutive_misses_before_reset:
                print("DBG: ROI Miss, Full Search on processed image")
            candidate_raw_corners_on_proc_img = self._find_board_contour_in_image(
                processed_image, proc_img_area, False, debug_mode,
                orig_img_dbg_for_raw_contour_window=processed_image if debug_mode else None)
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
                        if debug_mode: print(f"DBG: Smoothed (Drift: {avg_drift:.1f}px)")
                    else:
                        final_corners_on_proc_img = ordered_raw_corners
                        if debug_mode: print(f"DBG: Raw (Large Drift: {avg_drift:.1f}px)")
                else:
                    final_corners_on_proc_img = ordered_raw_corners
                    if debug_mode: print("DBG: Raw (No history)")
                if final_corners_on_proc_img is not None :
                    self.last_known_corners = final_corners_on_proc_img / scale_ratio if scale_ratio != 0 else final_corners_on_proc_img
                self.consecutive_misses = 0
        else:
            self.consecutive_misses += 1
            if last_known_corners_scaled is not None and self.consecutive_misses < self.max_consecutive_misses_before_reset:
                final_corners_on_proc_img = last_known_corners_scaled
                if debug_mode: print(f"DBG: Coasting (Miss #{self.consecutive_misses})")
            else:
                if debug_mode and self.last_known_corners is not None: print("DBG: Max misses, reset corners.")
                self.last_known_corners = None; final_corners_on_proc_img = None
        if final_corners_on_proc_img is None: return None, None
        final_corners_original_scale = final_corners_on_proc_img / scale_ratio if scale_ratio != 0 else final_corners_on_proc_img
        dst_warp = np.array([[0,0], [self.output_size-1,0], [self.output_size-1,self.output_size-1], [0,self.output_size-1]], dtype="float32")
        try:
            matrix = cv2.getPerspectiveTransform(final_corners_original_scale, dst_warp)
            warped_img = cv2.warpPerspective(image, matrix, (self.output_size, self.output_size))
        except cv2.error as e:
            if debug_mode: print(f"DBG: Warp Error: {e}, Corners: {final_corners_original_scale}")
            return None, final_corners_original_scale
        if debug_mode:
            dbg_disp_img = image.copy()
            if roi_was_used and 'roi_x1' in locals() and scale_ratio !=0:
                 roi_coords_orig = np.int0(np.array([roi_x1,roi_y1,roi_x2,roi_y2]) / scale_ratio)
                 cv2.rectangle(dbg_disp_img,(roi_coords_orig[0],roi_coords_orig[1]),(roi_coords_orig[2],roi_coords_orig[3]),(255,165,0),2)
            colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
            for i, p in enumerate(final_corners_original_scale):
                cv2.circle(dbg_disp_img, tuple(np.int0(p)), 7, colors[i], -1)
                cv2.putText(dbg_disp_img, str(i), tuple(np.int0(p)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            cv2.imshow("Debug - 06 Active Corners (on original img)", dbg_disp_img)
        return warped_img, final_corners_original_scale


# --- BoardStateIdentifier CLASS (NEW - VERY BASIC PLACEHOLDER) ---
class BoardStateIdentifier:
    def __init__(self, warped_board_size=480):
        self.warped_board_size = warped_board_size
        self.square_size = warped_board_size // 8
        # These thresholds are VERY naive and need calibration / a better method
        self.empty_intensity_threshold_low = 60  # Adjust based on your empty dark squares
        self.empty_intensity_threshold_high = 180 # Adjust based on your empty light squares
        self.piece_delta_threshold = 35 # Min diff from "empty" to be considered a piece
                                        # High values mean less sensitive to small changes / shadows

    def get_square_image(self, warped_board_img, rank, file): # rank 0-7 (chess 8-1), file 0-7 (chess a-h)
        """Extracts the image of a single square from the warped board."""
        if warped_board_img is None: return None
        y1 = rank * self.square_size
        y2 = (rank + 1) * self.square_size
        x1 = file * self.square_size
        x2 = (file + 1) * self.square_size
        square_img = warped_board_img[y1:y2, x1:x2]
        return square_img

    def identify_piece_on_square(self, square_img):
        """
        VERY NAIVE placeholder for piece identification.
        Returns: 'L' (Light piece), 'D' (Dark piece), 'E' (Empty)
        This needs to be replaced with a robust vision model.
        """
        if square_img is None or square_img.size == 0: return 'E'

        # Center crop to avoid edges of square influencing
        h, w = square_img.shape[:2]
        ch, cw = h // 4, w // 4 # crop 25% from each side
        cropped_sq_img = square_img[ch:h-ch, cw:w-cw]
        if cropped_sq_img.size == 0: cropped_sq_img = square_img # fallback

        gray_square = cv2.cvtColor(cropped_sq_img, cv2.COLOR_BGR2GRAY)
        avg_intensity = np.mean(gray_square)

        # Determine square color (very crudely - assumes distinct light/dark squares)
        is_light_square_expected = (cv2.mean(square_img)[0] + cv2.mean(square_img)[1] + cv2.mean(square_img)[2]) / 3 > 128 # Arbitrary
        # This is a placeholder: Real square color detection or prior knowledge is better

        # Super naive: If avg_intensity is far from what an empty square of that color would be
        if is_light_square_expected: # Expecting a light square
            if avg_intensity < self.empty_intensity_threshold_high - self.piece_delta_threshold:
                return 'D' # Dark piece on light square
        else: # Expecting a dark square
            if avg_intensity > self.empty_intensity_threshold_low + self.piece_delta_threshold:
                return 'L' # Light piece on dark square
        
        return 'E' # Otherwise, assume empty


    def get_board_state_map(self, warped_board_img):
        """Returns an 8x8 list of lists representing the board state (L, D, E)."""
        if warped_board_img is None:
            return [['E' for _ in range(8)] for _ in range(8)] # Empty board if no image

        board_map = []
        for r_idx in range(8): # Iterate 0-7 (chess ranks 8 to 1)
            row_map = []
            for f_idx in range(8): # Iterate 0-7 (chess files a to h)
                sq_img = self.get_square_image(warped_board_img, r_idx, f_idx)
                piece_symbol = self.identify_piece_on_square(sq_img)
                row_map.append(piece_symbol)
            board_map.append(row_map)
        return board_map

    def detect_move_from_maps(self, prev_map, current_map, chess_board_before_opponent_move, ai_color=chess.WHITE):
        """
        Detects a single move from two 8x8 board maps.
        Returns a chess.Move object or None.
        `chess_board_before_opponent_move` is the python-chess board state *before* the opponent made their physical move.
        `ai_color` is the color of your robot/AI. The opponent is the other color.
        """
        if prev_map == current_map:
            return None # No change detected

        opponent_color = not ai_color
        
        # Find differences
        changed_squares_from = [] # Squares that piece might have moved FROM
        changed_squares_to = []   # Squares that piece might have moved TO
        
        for r in range(8):
            for f in range(8):
                prev_piece = prev_map[r][f]
                curr_piece = current_map[r][f]
                
                if prev_piece != curr_piece:
                    # Square went from Occupied to Empty: candidate 'from_square'
                    if prev_piece != 'E' and curr_piece == 'E':
                        changed_squares_from.append(chess.square(f, 7 - r)) # chess.square uses 0-7 for file, 0-7 for rank (A1=0)
                    # Square went from Empty to Occupied, or piece type/color changed (capture)
                    elif curr_piece != 'E': # (prev_piece == 'E' and curr_piece != 'E') or (prev_piece != 'E' and curr_piece != 'E' and prev_piece != curr_piece)
                        changed_squares_to.append(chess.square(f, 7 - r))

        # Simplistic logic: Find the most plausible move from legal moves
        # This approach is more robust to noisy vision than just diffing counts.
        best_match_move = None
        min_mismatched_squares_after_move = 65 # More than possible squares

        if not chess_board_before_opponent_move.is_game_over():
            for legal_move in chess_board_before_opponent_move.legal_moves:
                # Create a hypothetical board state if this legal_move was made
                temp_board = chess_board_before_opponent_move.copy()
                temp_board.push(legal_move)
                
                # Convert this hypothetical python-chess board to our L/D/E map format
                hypothetical_map = [['E' for _ in range(8)] for _ in range(8)]
                for sq_idx in chess.SQUARES:
                    piece = temp_board.piece_at(sq_idx)
                    if piece:
                        r_map, f_map = 7 - chess.square_rank(sq_idx), chess.square_file(sq_idx)
                        # Our map's L/D depends on what `identify_piece_on_square` *would* return for that piece
                        # This is tricky without true piece ID. Let's simplify for the mock.
                        # If our identify_piece_on_square just distinguishes "my piece" vs "opponent piece" vs "empty",
                        # then we can use that. Assume 'L' for AI, 'D' for opponent, or vice-versa
                        hypothetical_map[r_map][f_map] = 'L' if piece.color == ai_color else 'D'

                # Compare this hypothetical_map with the current_map from vision
                mismatches = 0
                for r_ in range(8):
                    for f_ in range(8):
                        if hypothetical_map[r_][f_] != current_map[r_][f_]:
                            mismatches += 1
                
                if mismatches < min_mismatched_squares_after_move:
                    min_mismatched_squares_after_move = mismatches
                    best_match_move = legal_move
                # Add a small tolerance: if mismatches are very few (e.g. <=2 due to vision noise), consider it a good match
                elif mismatches <= 2 and best_match_move is None : # Prioritize some match over no match
                     min_mismatched_squares_after_move = mismatches
                     best_match_move = legal_move


        # If after checking all legal moves, the best match still has many mismatches,
        # it's probably not a valid detection. Threshold this.
        if min_mismatched_squares_after_move > 4 : # Arbitrary threshold, needs tuning based on vision reliability
            #print(f"DBG: Move detection uncertain, min_mismatches: {min_mismatched_squares_after_move}")
            return None
            
        return best_match_move


# --- Arm Controller Class (From previous version, with correction) ---
class ArmController:
    SERVO_CHANNELS = {'base':0,'shoulder':1,'elbow':2,'wrist_ud':3,'wrist_rot':4,'claw':5}
    SERVO_PARAMS = {
        'base': {'min_pulse':500,'max_pulse':2500,'default_angle':90,'min_angle':0,'max_angle':180,'current_angle':90},
        'shoulder': {'min_pulse':500,'max_pulse':2500,'default_angle':135,'min_angle':30,'max_angle':150,'current_angle':135},
        'elbow': {'min_pulse':500,'max_pulse':2500,'default_angle':45,'min_angle':0,'max_angle':135,'current_angle':45},
        'wrist_ud': {'min_pulse':500,'max_pulse':2500,'default_angle':90,'min_angle':0,'max_angle':180,'current_angle':90},
        'wrist_rot': {'min_pulse':500,'max_pulse':2500,'default_angle':90,'min_angle':0,'max_angle':180,'current_angle':90},
        'claw': {'min_pulse':600,'max_pulse':2400,'default_angle':45,'min_angle':45,'max_angle':110,'current_angle':45}
    }
    def __init__(self):
        self.servos={}; self.pca=None
        if not PCA_AVAILABLE: print("ArmController: PCA9685 libs not available. Arm disabled."); return
        try:
            i2c=busio.I2C(board.SCL,board.SDA); self.pca=PCA9685(i2c); self.pca.frequency=50
            print("PCA9685 initialized successfully.")
            for name,chan in self.SERVO_CHANNELS.items():
                if name not in self.SERVO_PARAMS: print(f"W: No params for servo '{name}'"); continue
                p=self.SERVO_PARAMS[name]
                s=adafruit_servo_motor.Servo(self.pca.channels[chan],min_pulse=p['min_pulse'],max_pulse=p['max_pulse'],actuation_range=180)
                self.servos[name]=s; p['current_angle']=p['default_angle']
                self.set_servo_angle(name,p['default_angle'],initial_setup=True); time.sleep(0.05)
            print("Servos initialized (attempted). VERIFY POSITIONS AND CALIBRATE!")
        except ValueError as e: print(f"CRIT: No PCA9685. Check I2C. {e}"); self.pca=None
        except Exception as e: print(f"CRIT: ArmController init err: {e}"); self.pca=None
    def set_servo_angle(self,name,angle,initial_setup=False):
        if not self.pca or name not in self.servos: return
        s,p=self.servos[name],self.SERVO_PARAMS[name]
        clamped=max(p['min_angle'],min(p['max_angle'],float(angle)))
        try: s.angle=clamped; p['current_angle']=clamped
        except Exception as e: print(f"Err servo '{name}': {e}")
    def go_to_home_position(self):
        if not self.pca: print("Arm not ready."); return
        print("Arm: Home..."); order=['elbow','shoulder','wrist_ud','wrist_rot','base','claw']
        for name in order:
            if name in self.SERVO_PARAMS: self.set_servo_angle(name,self.SERVO_PARAMS[name]['default_angle']); time.sleep(0.2)
        print("Arm: At home.")
    def operate_gripper(self,state,position=None):
        if not self.pca: print("Arm not ready."); return
        p=self.SERVO_PARAMS['claw']
        target=position if position is not None else (p['min_angle'] if state=="open" else (p['max_angle'] if state=="close" else None))
        if target is None: print(f"Gripper: Unknown state '{state}'."); return
        print(f"Gripper: {state.capitalize()}"); self.set_servo_angle('claw',target); time.sleep(0.3)
    def test_servo_increment(self,name,inc=5):
        if not self.pca or name not in self.SERVO_PARAMS: return
        curr=self.SERVO_PARAMS[name]['current_angle']; new=curr+inc
        print(f"Servo '{name}': {curr:.1f} -> {new:.1f}"); self.set_servo_angle(name,new)
    def test_servo_decrement(self,name,dec=5):
        if not self.pca or name not in self.SERVO_PARAMS: return
        curr=self.SERVO_PARAMS[name]['current_angle']; new=curr-dec
        print(f"Servo '{name}': {curr:.1f} -> {new:.1f}"); self.set_servo_angle(name,new)
    def print_servo_angles(self):
        if not self.pca: print("Arm not ready."); return
        print("\n--- Current Servo Angles ---")
        for name,p in self.SERVO_PARAMS.items(): print(f"S '{name}': {p['current_angle']:.1f}° (Min:{p['min_angle']},Max:{p['max_angle']},Def:{p['default_angle']})")
        print("---------------------------\nUpdate SERVO_PARAMS to save.")
    def move_piece_on_board(self,from_sq,to_sq,capture=False): # PLACEHOLDER
        if not self.pca: print("Arm not ready."); return
        print(f"ARM PLACEHOLDER: Move {from_sq} to {to_sq}{' (capture)' if capture else ''}")
        print("    TODO: Implement Board-to-Arm Coordinates, Inverse Kinematics, Path Planning.")
        time.sleep(1) # Simulate arm movement time
    def release_servos(self):
        if not self.pca: print("PCA not init."); return
        print("Releasing servos..."); [self.pca.channels[i].duty_cycle==0 for i in range(16)]
        print("Servos released.")

# --- Initialize Webcam (from previous) ---
cap = None; CAM_INDEX = -1; MAX_CAM_INDICES_TO_TRY = 4
for idx in range(MAX_CAM_INDICES_TO_TRY):
    cap_test=cv2.VideoCapture(idx)
    if cap_test.isOpened():
        cap,CAM_INDEX=cap_test,idx; cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        w,h=cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Webcam OK (idx {CAM_INDEX}, W:{w}, H:{h})."); break
    else: cap_test.release()
if cap is None: print("CRIT: Webcam init failed.")

# --- Load Stockfish Engine (from previous) ---
engine=None; engine_path="/usr/games/stockfish"
try:
    if not(os.path.exists(engine_path)and os.access(engine_path,os.X_OK)): raise FileNotFoundError("Stockfish missing/noexec")
    engine=chess.engine.SimpleEngine.popen_uci(engine_path); print(f"Stockfish OK: {engine_path}")
except Exception as e: print(f"Err Stockfish: {e}")

# --- Game State & Constants ---
current_chess_board = chess.Board()
AI_PLAYS_AS = chess.WHITE # Change this to chess.BLACK if you want AI to be black
human_player_turn = (AI_PLAYS_AS == chess.BLACK) # True if human starts (AI is black)

# --- Main Loop ---
def main():
    global current_chess_board, cap, CAM_INDEX, engine, human_player_turn, AI_PLAYS_AS

    if cap is None: print("EXIT: Webcam not available."); return

    board_detector = BoardDetector(output_size=480)
    state_identifier = BoardStateIdentifier(warped_board_size=board_detector.output_size)
    arm_controller = ArmController()

    RUN_IN_DEBUG_MODE = False
    print_keys_help()

    active_test_servo = 'base'
    if arm_controller.pca is None: print("INFO: ARM DISABLED (PCA/libs issue).")

    # For move detection
    previous_board_map = None # Stores the 8x8 L/D/E map from vision
    waiting_for_human_move = human_player_turn # Start waiting if it's human's turn

    LOG_PERFORMANCE = False; loop_time_filter = []; start_time_perf = 0

    game_phase = "AI_THINKING" if not human_player_turn else "HUMAN_MOVE_WAIT"
    print(f"--- Game Start --- AI plays as {'White' if AI_PLAYS_AS == chess.WHITE else 'Black'}. Current phase: {game_phase}")


    while True:
        if LOG_PERFORMANCE: start_time_perf = time.monotonic()
        ret, frame = cap.read()
        if not ret or frame is None:
            if handle_camera_failure(): break # True if critical
            continue

        # --- Board Outline Detection ---
        processed_for_detector = frame.copy()
        warped_board_img, board_corners = board_detector.detect(processed_for_detector, debug_mode=RUN_IN_DEBUG_MODE)

        # --- Display and UI ---
        display_frame = frame.copy()
        if warped_board_img is not None:
            cv2.imshow("1. Warped Board", warped_board_img)
            if board_corners is not None: cv2.polylines(display_frame, [np.int32(board_corners)], True, (0,255,0), 2, cv2.LINE_AA)
            if RUN_IN_DEBUG_MODE and waiting_for_human_move: # Show current state map if debugging opponent move
                debug_current_map = state_identifier.get_board_state_map(warped_board_img)
                # print("Debug Current Visual Map:", debug_current_map) # Can be spammy
        else:
            display_placeholder_board(board_detector.output_size)

        update_display_overlays(display_frame, active_test_servo, LOG_PERFORMANCE, loop_time_filter, start_time_perf if LOG_PERFORMANCE else 0, game_phase)
        cv2.imshow("3. Live Feed", display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        # --- Key Handling ---
        RUN_IN_DEBUG_MODE, _, active_test_servo, should_quit = \
            handle_key_presses(key, RUN_IN_DEBUG_MODE, False, active_test_servo,
                               current_chess_board, board_detector, arm_controller, game_phase) # engine_play_requested not used here
        if should_quit: break


        # --- Game Logic FSM ---
        if current_chess_board.is_game_over():
            game_phase = "GAME_OVER"
            # Handled at loop end / cleanup

        elif game_phase == "AI_THINKING":
            if engine and arm_controller.pca:
                print(f"\nPhase: {game_phase} (AI is {chess.COLOR_NAMES[AI_PLAYS_AS]})")
                time.sleep(0.1) # UI to update
                try:
                    result = engine.play(current_chess_board, chess.engine.Limit(time=0.5, depth=8)) # More thinking time
                    if result.move:
                        print(f"AI wants to play: {result.move.uci()}")
                        arm_controller.move_piece_on_board(
                            result.move.uci()[:2], result.move.uci()[2:],
                            current_chess_board.is_capture(result.move)
                        )
                        current_chess_board.push(result.move)
                        print("Current board (after AI move):")
                        print(current_chess_board)
                        if warped_board_img is not None: # Update expected visual state
                            previous_board_map = state_identifier.get_board_state_map(warped_board_img)
                        game_phase = "HUMAN_MOVE_WAIT"
                        waiting_for_human_move = True
                    else:
                        print("AI engine found no move. This might be a bug or stalemate/checkmate.")
                        game_phase = "GAME_OVER" # Or handle error
                except Exception as e:
                    print(f"Engine play error: {e}")
                    game_phase = "ERROR" # Or retry
            else: # AI can't play (no engine or no arm)
                # This state means it's AI's turn but it cannot act.
                # If game is not over, this is effectively a stall. For this demo, we wait for 'm' key.
                print(f"AI Turn, but Engine/Arm not ready. Press 'm' to try manual AI move or 's' to switch to human turn.")
                if key == ord('s'): # Manual override to switch turn
                     human_player_turn = True
                     waiting_for_human_move = True
                     game_phase = "HUMAN_MOVE_WAIT"
                     print("Manually switched to Human's turn.")
                elif key == ord('m') and engine and arm_controller.pca: # Try manual trigger if 'm' is pressed
                    # This block is similar to above, just for manual trigger if FSM gets stuck
                    print(f"Manual AI Move Triggered...")
                    # (Duplicate code here for brevity, ideally refactor AI move logic)
                    try:
                        result = engine.play(current_chess_board, chess.engine.Limit(time=0.3, depth=6))
                        if result.move:
                            print(f"AI wants to play: {result.move.uci()}")
                            arm_controller.move_piece_on_board(
                                result.move.uci()[:2], result.move.uci()[2:],
                                current_chess_board.is_capture(result.move)
                            )
                            current_chess_board.push(result.move)
                            print("Current board (after AI move):")
                            print(current_chess_board)
                            if warped_board_img is not None:
                                previous_board_map = state_identifier.get_board_state_map(warped_board_img)
                            game_phase = "HUMAN_MOVE_WAIT"
                            waiting_for_human_move = True
                        else: print("AI engine found no move.")
                    except Exception as e: print(f"Engine play error: {e}")


        elif game_phase == "HUMAN_MOVE_WAIT":
            if not waiting_for_human_move: # Should not happen if logic is correct
                game_phase = "AI_THINKING" # Back to AI
                continue

            #print(f"\rPhase: {game_phase} ({chess.COLOR_NAMES[not AI_PLAYS_AS]}'s turn). Waiting for physical move...", end="")
            if warped_board_img is not None and previous_board_map is not None:
                current_visual_map = state_identifier.get_board_state_map(warped_board_img)
                
                if current_visual_map != previous_board_map : # Potential move
                    print("\nChange detected in visual board state!")
                    # Pass the chess.Board state *before* the human made their physical move
                    detected_move = state_identifier.detect_move_from_maps(
                        previous_board_map, current_visual_map, current_chess_board, AI_PLAYS_AS
                    )
                    
                    if detected_move:
                        if detected_move in current_chess_board.legal_moves:
                            print(f"Human opponent move DETECTED and LEGAL: {detected_move.uci()}")
                            current_chess_board.push(detected_move)
                            print("Current board (after human move):")
                            print(current_chess_board)
                            waiting_for_human_move = False # Human move processed
                            previous_board_map = current_visual_map # Update baseline
                            game_phase = "AI_THINKING"
                        else:
                            print(f"Human opponent move DETECTED BUT ILLEGAL: {detected_move.uci()}. Reverting visual map.")
                            # This implies vision might be noisy or an illegal physical move was made.
                            # For now, we revert the previous_board_map to current_visual_map to stabilize,
                            # but a more robust system might ask for the move to be corrected.
                            previous_board_map = current_visual_map # Re-baseline to current messy state
                    else:
                        # No clear move detected, could be noise. Update previous_board_map to current.
                        # This helps if a piece was just nudged.
                        # print("No single clear move detected from vision diff. Updating visual baseline.")
                        previous_board_map = current_visual_map
            elif warped_board_img is not None and previous_board_map is None and not current_chess_board.move_stack: # Initial state capture
                # Only capture initial state if it's truly the start (no moves made)
                # and it's human's turn to start.
                if human_player_turn:
                     previous_board_map = state_identifier.get_board_state_map(warped_board_img)
                     print("Initial board state captured for human's first move.")


        if game_phase == "GAME_OVER" and not RUN_IN_DEBUG_MODE:
            print("Game Over."); time.sleep(2) # Show message
            # break # Loop continues to allow key presses like 'r' or 'q'

    cleanup_resources(arm_controller, engine, cap)
    print_game_result(current_chess_board)

def print_keys_help():
    print("\n--- Chess Vision Arm Control ---")
    print("  'q':Quit | 'd':Toggle BoardDetect Debug | 's':Switch Turn (if AI stuck)")
    print("  Game:      'm':Manual AI Move Trigger (if stuck) | 'u':Undo | 'r':Reset") # 'm' functionality changed
    print("  Arm:  'h':Home | 'o':Open | 'c':Close | 'l':Release | 'p':Print Angles")
    print("        '1-6':Select Servo | '+/-':Increment/Decrement")

def handle_camera_failure():
    global cap, CAM_INDEX
    print("ERR: Cam fail. Retrying..."); time.sleep(0.1)
    if cap is not None: cap.release()
    cap=None; time.sleep(0.5)
    if CAM_INDEX!=-1:
        tmp=cv2.VideoCapture(CAM_INDEX)
        if tmp.isOpened(): cap=tmp; print("INFO: Cam re-opened."); cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480); return False
        else: print(f"CRIT: Fail re-open @{CAM_INDEX}."); tmp.release(); return True
    else: print("CRIT: Cam idx unknown."); return True

def display_placeholder_board(output_size):
    dummy=np.zeros((output_size,output_size,3),dtype=np.uint8)
    cv2.putText(dummy,"No Board",(20,output_size//2-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    cv2.imshow("1. Warped Board",dummy)

def update_display_overlays(display_frame, active_test_servo, log_perf, time_filter, start_t, game_phase_str):
    cv2.putText(display_frame, f"Servo:{active_test_servo.upper()} Phase:{game_phase_str}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
    cv2.putText(display_frame, "Keys:q,d,s|Game:m,u,r|Arm:h,o,c,l,p,1-6,+-", (10,display_frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,50),1,cv2.LINE_AA)
    if log_perf and start_t > 0:
        proc_time=time.monotonic()-start_t; time_filter.append(proc_time)
        if len(time_filter)>20: time_filter.pop(0)
        avg_fps=1.0/(sum(time_filter)/len(time_filter)) if time_filter else 0
        cv2.putText(display_frame,f"FPS:{avg_fps:.1f}",(display_frame.shape[1]-100,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1,cv2.LINE_AA)

def handle_key_presses(key, run_debug, eng_req, act_servo, board_state, board_det, arm_ctrl, game_phase_ref): # Pass game_phase by mutable ref if needed or return it
    # Note: 'm' for engine move is now handled by the FSM in main loop when it's AI's turn or manually via new key.
    # 'engine_play_requested' is therefore not directly set here for that purpose.
    global human_player_turn, game_phase # Allow modification of these global game state vars
    should_quit_flag = False
    can_arm = arm_ctrl.pca is not None

    if key == ord('q'): should_quit_flag = True; print("Quitting.")
    elif key == ord('d'):
        run_debug = not run_debug
        print(f"BoardDetector Debug Mode: {'ON' if run_debug else 'OFF'}")
        if not run_debug:
            for win_name in ["Debug - ROI Canny Edges","Debug - Full Canny Edges","Debug - 05 Raw Contour (on processed img)","Debug - 06 Active Corners (on original img)"]:
                try: cv2.destroyWindow(win_name)
                except cv2.error: pass
    # 'm' is now for *manual trigger* if AI is stuck (see main FSM)
    #elif key == ord('m'): # This 'm' is removed as primary trigger for AI move.
    #    if can_arm and engine: eng_req = True; print("Engine move requested via key.")
    #    # ... (other conditions for 'm')
    elif key == ord('u'): # Undo
        if board_state.move_stack: board_state.pop() # Opponent / last move
        if board_state.move_stack: board_state.pop() # AI / previous move
        print("Undo (2 moves)."); print(board_state)
        # Reset game phase to figure out whose turn it is now
        current_turn_is_ai = (board_state.turn == AI_PLAYS_AS)
        game_phase = "AI_THINKING" if current_turn_is_ai else "HUMAN_MOVE_WAIT"
        human_player_turn = not current_turn_is_ai # if AI's turn, human is not playing
        waiting_for_human_move = human_player_turn
        print(f"Board reverted. New phase: {game_phase}")


    elif key == ord('r'): # Reset
        board_state.reset(); board_det.last_known_corners=None; board_det.consecutive_misses=0
        if can_arm: arm_ctrl.go_to_home_position()
        human_player_turn = (AI_PLAYS_AS == chess.BLACK)
        waiting_for_human_move = human_player_turn
        game_phase = "AI_THINKING" if not human_player_turn else "HUMAN_MOVE_WAIT"
        previous_board_map = None # Reset visual baseline
        print(f"Board & Arm Reset. New phase: {game_phase}"); print(board_state)

    elif key == ord('h') and can_arm: arm_ctrl.go_to_home_position()
    elif key == ord('o') and can_arm: arm_ctrl.operate_gripper("open")
    elif key == ord('c') and can_arm: arm_ctrl.operate_gripper("close")
    elif key == ord('l') and can_arm: arm_ctrl.release_servos()
    elif key == ord('p') and can_arm: arm_ctrl.print_servo_angles()
    elif ord('1') <= key <= ord('6'):
        servo_map = {'1':'base','2':'shoulder','3':'elbow','4':'wrist_ud','5':'wrist_rot','6':'claw'}
        s_key = chr(key)
        if s_key in servo_map: act_servo = servo_map[s_key]; print(f"Sel Servo: {act_servo.upper()}")
    elif (key==ord('+') or key==ord('=')) and can_arm: arm_ctrl.test_servo_increment(act_servo,5)
    elif key==ord('-') and can_arm: arm_ctrl.test_servo_decrement(act_servo,5)
    elif key == ord('s'): # Switch turn manually (if stuck)
        print("Manual turn switch requested.")
        if game_phase == "AI_THINKING":
            game_phase = "HUMAN_MOVE_WAIT"
            human_player_turn = True
            waiting_for_human_move = True
        elif game_phase == "HUMAN_MOVE_WAIT":
            game_phase = "AI_THINKING"
            human_player_turn = False
            waiting_for_human_move = False
        print(f"Game phase manually set to: {game_phase}")


    return run_debug, eng_req, act_servo, should_quit_flag

def cleanup_resources(arm_controller, stockfish_engine, camera_capture):
    print("Cleaning up...");
    if arm_controller.pca is not None: arm_controller.release_servos()
    if stockfish_engine:
        try: stockfish_engine.quit()
        except: pass
    if camera_capture: camera_capture.release()
    cv2.destroyAllWindows()

def print_game_result(board_state):
    if board_state.is_checkmate(): print(f"Checkmate! {'White' if not board_state.turn else 'Black'} wins.")
    elif board_state.is_stalemate(): print("Stalemate! Draw.")
    # Add other draw conditions

if __name__ == "__main__":
    main()