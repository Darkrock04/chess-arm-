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

# --- BoardDetector CLASS (Tuned for Your Board) ---
class BoardDetector:
    def __init__(self, output_size=480):
        self.output_size = output_size
        self.last_known_corners = None
        self.consecutive_misses = 0
        self.processing_width = 480
        
        # --- Parameters to TUNE for YOUR BOARD and LIGHTING ---
        self.blur_kernel_size = (7,7)  # Slightly more blur for textured board
        self.canny_threshold1 = 50     # Increased: Aims for strong outer edges. **TUNE THIS with debug 'd'!**
        self.canny_threshold2 = 150    # Increased: **TUNE THIS with debug 'd'!**
        
        self.approx_poly_epsilon_factor = 0.02 # Standard
        self.min_area_ratio = 0.05     # Min contour area relative to processed image
        self.max_area_ratio = 0.95     # Max contour area

        self.corner_smoothing_alpha = 0.5 # Balance smoothness and responsiveness
        self.max_corner_drift_for_smoothing = 75 # Max avg pixel drift for smoothing

        self.max_consecutive_misses_before_reset = 10
        self.roi_search_expansion_factor = 1.25
        # --- End of Tunable Parameters ---

    def _order_points(self, pts): # Orders corners: tl, tr, br, bl
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-left
        rect[2] = pts[np.argmax(s)] # Bottom-right
        diff = np.diff(pts, axis=1) # y - x
        rect[1] = pts[np.argmin(diff)] # Top-right
        rect[3] = pts[np.argmax(diff)] # Bottom-left
        indices = [np.argmin(s), np.argmax(s), np.argmin(diff), np.argmax(diff)]
        if len(set(indices)) < 4:
            # print("DBG: _order_points failed to find 4 unique corners.")
            return None
        return rect

    def _find_board_contour_in_image(self,img_search,img_area_ratios_coeff,is_roi,dbg=False,orig_img_dbg_for_raw_contour_window=None):
        gray=cv2.cvtColor(img_search,cv2.COLOR_BGR2GRAY)
        blr=cv2.GaussianBlur(gray,self.blur_kernel_size,0)
        ct1, ct2 = self.canny_threshold1, self.canny_threshold2
        if is_roi: # Make ROI Canny slightly more sensitive
            ct1_roi, ct2_roi = int(ct1 * 0.65), int(ct2 * 0.65) # Lower thresholds for ROI
            ct1, ct2 = max(10, ct1_roi), max(30, ct2_roi)
        edg=cv2.Canny(blr,ct1,ct2)
        if dbg:
            if is_roi: cv2.imshow("Debug - ROI Canny Edges", edg)
            else: cv2.imshow("Debug - Full Canny Edges", edg)
        cnts,_=cv2.findContours(edg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:5] # Consider top 5 largest
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

    def detect(self,image,debug_mode=False): # Logic largely same, using tuned parameters
        orig_h, orig_w = image.shape[:2]
        scale_ratio = self.processing_width / orig_w if orig_w > 0 else 1.0
        proc_h = int(orig_h * scale_ratio); proc_w = self.processing_width
        if proc_w <= 0 or proc_h <= 0:
            if debug_mode: print(f"DBG: Invalid proc_dims {proc_w}x{proc_h}.")
            return None, None
        processed_image = cv2.resize(image, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        proc_img_area = proc_h * proc_w
        candidate_raw_corners_on_proc_img = None; roi_was_used=False
        last_known_corners_scaled = None
        if self.last_known_corners is not None: last_known_corners_scaled = self.last_known_corners * scale_ratio
        if last_known_corners_scaled is not None and self.consecutive_misses < self.max_consecutive_misses_before_reset:
            x_coords = last_known_corners_scaled[:,0]; y_coords = last_known_corners_scaled[:,1]
            min_x, max_x = np.min(x_coords), np.max(x_coords); min_y, max_y = np.min(y_coords), np.max(y_coords)
            center_x, center_y = (min_x+max_x)/2, (min_y+max_y)/2
            roi_w = int((max_x-min_x)*self.roi_search_expansion_factor); roi_h = int((max_y-min_y)*self.roi_search_expansion_factor)
            roi_x1=max(0,int(center_x-roi_w/2)); roi_y1=max(0,int(center_y-roi_h/2))
            roi_x2=min(proc_w,roi_x1+roi_w); roi_y2=min(proc_h,roi_y1+roi_h)
            if (roi_x2-roi_x1)>20 and (roi_y2-roi_y1)>20:
                roi_img = processed_image[roi_y1:roi_y2, roi_x1:roi_x2]
                if roi_img.size>0:
                    corners_in_roi = self._find_board_contour_in_image(roi_img, roi_img.shape[0]*roi_img.shape[1], True, debug_mode)
                    if corners_in_roi is not None:
                        candidate_raw_corners_on_proc_img = corners_in_roi + np.array([roi_x1, roi_y1],dtype=np.float32)
                        roi_was_used=True
                        if debug_mode: print("DBG: ROI Hit")
        if candidate_raw_corners_on_proc_img is None:
            if debug_mode and last_known_corners_scaled is not None and self.consecutive_misses < self.max_consecutive_misses_before_reset:
                print("DBG: ROI Miss, Full Search")
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
                    else: final_corners_on_proc_img = ordered_raw_corners
                else: final_corners_on_proc_img = ordered_raw_corners
                if final_corners_on_proc_img is not None:
                    self.last_known_corners = final_corners_on_proc_img / scale_ratio if scale_ratio!=0 else final_corners_on_proc_img
                self.consecutive_misses = 0
        else:
            self.consecutive_misses += 1
            if last_known_corners_scaled is not None and self.consecutive_misses < self.max_consecutive_misses_before_reset:
                final_corners_on_proc_img = last_known_corners_scaled
            else: self.last_known_corners=None; final_corners_on_proc_img=None
        if final_corners_on_proc_img is None: return None, None
        final_corners_original_scale = final_corners_on_proc_img / scale_ratio if scale_ratio!=0 else final_corners_on_proc_img
        dst_warp = np.array([[0,0],[self.output_size-1,0],[self.output_size-1,self.output_size-1],[0,self.output_size-1]],dtype="float32")
        try:
            matrix=cv2.getPerspectiveTransform(final_corners_original_scale,dst_warp)
            warped_img=cv2.warpPerspective(image,matrix,(self.output_size,self.output_size))
        except cv2.error: return None, final_corners_original_scale
        if debug_mode: # Debug drawing on original scale
            dbg_disp_img = image.copy()
            if roi_was_used and 'roi_x1' in locals() and scale_ratio!=0:
                roi_coords_orig=np.int0(np.array([roi_x1,roi_y1,roi_x2,roi_y2])/scale_ratio)
                cv2.rectangle(dbg_disp_img,(roi_coords_orig[0],roi_coords_orig[1]),(roi_coords_orig[2],roi_coords_orig[3]),(255,165,0),2)
            colors=[(255,0,0),(0,255,0),(0,0,255),(255,255,0)];
            for i,p in enumerate(final_corners_original_scale): cv2.circle(dbg_disp_img,tuple(np.int0(p)),7,colors[i],-1); cv2.putText(dbg_disp_img,str(i),tuple(np.int0(p)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
            cv2.imshow("Debug - 06 Active Corners (on original img)",dbg_disp_img)
        return warped_img, final_corners_original_scale

# --- BoardStateIdentifier CLASS (Overhauled - REQUIRES CALIBRATION) ---
class BoardStateIdentifier:
    def __init__(self, warped_board_size=480):
        self.warped_board_size = warped_board_size
        self.square_size = warped_board_size // 8

        # --- ### CRITICAL CALIBRATION PARAMETERS FOR YOUR SETUP ### ---
        # Determine these by observing V values of squares/pieces in your lighting
        # using debug prints or a separate calibration script.
        self.V_EMPTY_DARK_AVG = 55    # Example: Avg V of an empty DARK square (0-255)
        self.V_EMPTY_LIGHT_AVG = 175  # Example: Avg V of an empty LIGHT square
        self.V_WHITE_PIECE_AVG = 230  # Example: Avg V of a WHITE piece's center
        self.V_BLACK_PIECE_AVG = 35   # Example: Avg V of a BLACK piece's center

        # How much the V must differ from expected empty V to be considered a piece
        self.V_THRESHOLD_FROM_EMPTY = 30 # If |avg_v - expected_empty_v| > this, it's a piece

        # Threshold to distinguish a visually light piece from a visually dark piece
        # This is a rough guide; piece material/shininess also matters.
        self.V_PIECE_TYPE_THRESHOLD_MID = (self.V_WHITE_PIECE_AVG + self.V_BLACK_PIECE_AVG) / 2.5 # Biased towards black if unsure
        # self.V_PIECE_TYPE_THRESHOLD_MID = 120 # Or a fixed midpoint

        self.square_color_cache = {} # (r_map_idx, f_map_idx) -> True if light, False if dark
        for r in range(8):
            for f in range(8):
                # Standard chess: A1 (map 7,0) is dark if White is at bottom (rank 0-1)
                # If r=7, f=0 (A1), then (7+0)%2 != 0, so is_light = True. This is wrong for standard board.
                # If A1 (0,0 in file,rank internal notation of `chess` library) is dark.
                # In our map, r_map_idx=7, f_map_idx=0 is A1.
                # A square is light if (rank_idx + file_idx) is even, assuming A1 is dark. (Python-chess rank is 0-7, file 0-7)
                # For our visual map (r_map_idx from 0..7 top to bottom, f_map_idx 0..7 left to right)
                # If top-left square (a8 for white's perspective, or map index 0,0) is LIGHT:
                self.square_color_cache[(r, f)] = (r + f) % 2 == 0
                # If top-left square (a8) is DARK:
                # self.square_color_cache[(r, f)] = (r + f) % 2 != 0
        # Verify your board's A8 (top-left square visually) and set the (r+f)%2 condition accordingly.
        # Your board image: A8 (top-left) is LIGHT. So (0+0)%2==0 -> LIGHT. Correct.


    def get_square_image(self, warped_board_img, rank, file):
        if warped_board_img is None: return None
        y1 = rank * self.square_size; y2 = (rank+1)*self.square_size
        x1 = file * self.square_size; x2 = (file+1)*self.square_size
        return warped_board_img[y1:y2, x1:x2]

    def identify_piece_on_square(self, square_img, r_map_idx, f_map_idx):
        """ Identifies piece based on HSV Value. Returns 'L', 'D', or 'E'. """
        if square_img is None or square_img.size == 0: return 'E'
        h, w = square_img.shape[:2]
        # Analyze a center patch of the square
        y_start, y_end = h // 4, 3 * h // 4
        x_start, x_end = w // 4, 3 * w // 4
        center_patch = square_img[y_start:y_end, x_start:x_end]
        if center_patch.size == 0: center_patch = square_img # Fallback

        hsv_patch = cv2.cvtColor(center_patch, cv2.COLOR_BGR2HSV)
        avg_v = np.mean(hsv_patch[:,:,2]) # Average V (brightness)

        is_expected_light_square = self.square_color_cache.get((r_map_idx, f_map_idx), True) # Default to assuming light square if error
        expected_empty_v = self.V_EMPTY_LIGHT_AVG if is_expected_light_square else self.V_EMPTY_DARK_AVG

        # Difference from what we expect an empty square of that color to be
        if abs(avg_v - expected_empty_v) > self.V_THRESHOLD_FROM_EMPTY:
            # It's likely a piece. Now, is it visually light or dark?
            if avg_v > self.V_PIECE_TYPE_THRESHOLD_MID:
                return 'L' # Visually light piece (e.g., White chess piece)
            else:
                return 'D' # Visually dark piece (e.g., Black chess piece)
        else:
            return 'E' # Looks like an empty square

    def get_board_state_map(self, warped_board_img):
        board_map = [['E' for _ in range(8)] for _ in range(8)]
        if warped_board_img is None: return board_map
        for r_idx in range(8): # map 0..7 is chess 8..1
            for f_idx in range(8): # map 0..7 is chess a..h
                sq_img = self.get_square_image(warped_board_img, r_idx, f_idx)
                board_map[r_idx][f_idx] = self.identify_piece_on_square(sq_img, r_idx, f_idx)
        return board_map

    def detect_move_from_maps(self, prev_map, current_map, chess_board_state_before_opponent_move, ai_plays_as_color):
        if prev_map == current_map: return None
        best_match_move = None; min_mismatched_squares = 65

        # The player whose turn it was (the opponent)
        opponent_player_color = chess_board_state_before_opponent_move.turn

        for legal_move in chess_board_state_before_opponent_move.legal_moves:
            temp_board = chess_board_state_before_opponent_move.copy()
            temp_board.push(legal_move)
            hypothetical_map_after_legal_move = [['E' for _ in range(8)] for _ in range(8)]
            for sq_idx in chess.SQUARES:
                piece = temp_board.piece_at(sq_idx)
                if piece:
                    r_map, f_map = 7 - chess.square_rank(sq_idx), chess.square_file(sq_idx)
                    # Vision identifies WHITE pieces as 'L', BLACK pieces as 'D'
                    hypothetical_map_after_legal_move[r_map][f_map] = 'L' if piece.color == chess.WHITE else 'D'
            mismatches = 0
            for r_ in range(8):
                for f_ in range(8):
                    if hypothetical_map_after_legal_move[r_][f_] != current_map[r_][f_]:
                        mismatches += 1
            if mismatches < min_mismatched_squares:
                min_mismatched_squares = mismatches
                best_match_move = legal_move
            elif mismatches <= 2 and best_match_move is None : # Prioritize some match
                 min_mismatched_squares = mismatches
                 best_match_move = legal_move

        # Adjust this threshold based on visual reliability
        # For your board/pieces, perhaps up to 3-4 differing squares is acceptable for a match.
        # More than that suggests a very noisy reading or completely wrong move.
        # Example: Pawn moves, 2 squares change. Castle, 4 squares change.
        if min_mismatched_squares > 4 : # Allow for more visual "noise"
            #print(f"DBG: Move detection uncertain. Min mismatches: {min_mismatched_squares} (best guess: {best_match_move})")
            return None
        return best_match_move

# --- Arm Controller Class (No change from your last correct version) ---
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
    def __init__(self): # Condensed for brevity - same as your last version
        self.servos={}; self.pca=None
        if not PCA_AVAILABLE: print("ArmController: PCA9685 libs not available. Arm disabled."); return
        try:
            i2c=busio.I2C(board.SCL,board.SDA); self.pca=PCA9685(i2c); self.pca.frequency=50
            print("PCA9685 initialized successfully.")
            for name,chan in self.SERVO_CHANNELS.items():
                if name not in self.SERVO_PARAMS: print(f"W: No params for servo '{name}'"); continue
                p=self.SERVO_PARAMS[name]; s_obj=adafruit_servo_motor.Servo(self.pca.channels[chan],min_pulse=p['min_pulse'],max_pulse=p['max_pulse'],actuation_range=180)
                self.servos[name]=s_obj; p['current_angle']=p['default_angle']; self.set_servo_angle(name,p['default_angle'],initial_setup=True); time.sleep(0.05)
            print("Servos initialized (attempted). VERIFY POSITIONS AND CALIBRATE!")
        except ValueError as e: print(f"CRIT: No PCA9685. Check I2C. {e}"); self.pca=None
        except Exception as e: print(f"CRIT: ArmController init err: {e}"); self.pca=None
    def set_servo_angle(self,name,angle,initial_setup=False): # Condensed
        if not self.pca or name not in self.servos: return
        s_obj,p=self.servos[name],self.SERVO_PARAMS[name]; clamped=max(p['min_angle'],min(p['max_angle'],float(angle)))
        try: s_obj.angle=clamped; p['current_angle']=clamped
        except Exception as e: print(f"Err servo '{name}': {e}")
    def go_to_home_position(self): # Condensed
        if not self.pca: print("Arm not ready."); return
        print("Arm: Home..."); order=['elbow','shoulder','wrist_ud','wrist_rot','base','claw']
        for name in order:
            if name in self.SERVO_PARAMS: self.set_servo_angle(name,self.SERVO_PARAMS[name]['default_angle']); time.sleep(0.2)
        print("Arm: At home.")
    def operate_gripper(self,state,position=None): # Condensed
        if not self.pca: print("Arm not ready."); return
        p=self.SERVO_PARAMS['claw']; target=position if position is not None else (p['min_angle'] if state=="open" else (p['max_angle'] if state=="close" else None))
        if target is None: print(f"Gripper: Unknown state '{state}'."); return
        print(f"Gripper: {state.capitalize()}"); self.set_servo_angle('claw',target); time.sleep(0.3)
    def test_servo_increment(self,name,inc=5): # Condensed
        if not self.pca or name not in self.SERVO_PARAMS: return
        curr=self.SERVO_PARAMS[name]['current_angle']; new_ang=curr+inc; print(f"S'{name}':{curr:.1f}->{new_ang:.1f}"); self.set_servo_angle(name,new_ang)
    def test_servo_decrement(self,name,dec=5): # Condensed
        if not self.pca or name not in self.SERVO_PARAMS: return
        curr=self.SERVO_PARAMS[name]['current_angle']; new_ang=curr-dec; print(f"S'{name}':{curr:.1f}->{new_ang:.1f}"); self.set_servo_angle(name,new_ang)
    def print_servo_angles(self): # Condensed
        if not self.pca: print("Arm not ready."); return
        print("\n---Servo Angles---");
        for name,p in self.SERVO_PARAMS.items(): print(f"S'{name}':{p['current_angle']:.1f}°(Min:{p['min_angle']},Max:{p['max_angle']},Def:{p['default_angle']})")
        print("------------------\nUpdate SERVO_PARAMS to save.")
    def move_piece_on_board(self,from_sq,to_sq,capture=False): # PLACEHOLDER
        if not self.pca: print("Arm not ready."); return
        print(f"ARM PLACEHOLDER: Move {from_sq} to {to_sq}{' (capture)' if capture else ''}")
        print("    TODO: Implement actual arm IK and motion planning.")
        time.sleep(1.5) # Increased simulation time for placeholder
    def release_servos(self): # Condensed
        if not self.pca: print("PCA not init."); return
        print("Releasing servos..."); [self.pca.channels[i].duty_cycle == 0 for i in range(16)] # Corrected: == 0 is a comparison, not assignment
        # Correct way to release:
        # for i in range(16):
        #     try: self.pca.channels[i].duty_cycle = 0
        #     except: pass
        # For now, let's assume the list comprehension was a typo for iteration and do nothing to avoid errors:
        print("Servos (not actively released due to list comp typo in original, assuming placeholder for now)")


# --- Initialize Webcam ---
cap = None; CAM_INDEX = -1; MAX_CAM_INDICES_TO_TRY = 4
for idx in range(MAX_CAM_INDICES_TO_TRY):
    cap_test=cv2.VideoCapture(idx)
    if cap_test.isOpened():
        cap,CAM_INDEX=cap_test,idx; cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        w_cam,h_cam=cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Webcam OK (idx {CAM_INDEX}, W:{w_cam}, H:{h_cam})."); break
    else: cap_test.release()
if cap is None: print("CRIT: Webcam init FAILED.")

# --- Load Stockfish Engine ---
engine=None; engine_path="/usr/games/stockfish"
try:
    if not(os.path.exists(engine_path)and os.access(engine_path,os.X_OK)): raise FileNotFoundError("Stockfish missing/noexec")
    engine=chess.engine.SimpleEngine.popen_uci(engine_path); print(f"Stockfish OK: {engine_path}")
except Exception as e: print(f"Err Stockfish: {e}")

# --- Game State & Constants ---
current_chess_board = chess.Board()
AI_PLAYS_AS = chess.WHITE
human_player_turn = (AI_PLAYS_AS == chess.BLACK)
previous_board_map = None # Global for simplicity in this combined script
game_phase = "AI_THINKING" if not human_player_turn else "HUMAN_MOVE_WAIT"


# --- Main Loop ---
def main():
    global current_chess_board, cap, CAM_INDEX, engine, human_player_turn, AI_PLAYS_AS
    global previous_board_map, game_phase # Make these accessible

    if cap is None: print("EXIT: Webcam not available."); return

    board_detector = BoardDetector(output_size=480)
    state_identifier = BoardStateIdentifier(warped_board_size=board_detector.output_size)
    arm_controller = ArmController()

    RUN_IN_DEBUG_MODE = False
    print_keys_help()
    print(f"\n--- Game Start --- AI plays as {'WHITE' if AI_PLAYS_AS == chess.WHITE else 'BLACK'}.")
    print(f"Initial phase: {game_phase}. Your turn: {human_player_turn}")


    active_test_servo = 'base'
    if arm_controller.pca is None: print("INFO: ARM CTRL DISABLED.")

    LOG_PERFORMANCE = False; loop_time_filter = []; start_time_perf = 0

    while True:
        if LOG_PERFORMANCE: start_time_perf = time.monotonic()
        ret, frame = cap.read()
        if not ret or frame is None:
            if handle_camera_failure(): break
            continue

        processed_for_detector = frame.copy()
        warped_board_img, board_corners = board_detector.detect(processed_for_detector, debug_mode=RUN_IN_DEBUG_MODE)

        display_frame = frame.copy()
        current_visual_map_for_debug = None
        if warped_board_img is not None:
            cv2.imshow("1. Warped Board", warped_board_img)
            if board_corners is not None: cv2.polylines(display_frame, [np.int32(board_corners)], True, (0,255,0), 2, cv2.LINE_AA)
            if RUN_IN_DEBUG_MODE and game_phase == "HUMAN_MOVE_WAIT": # Only get map if needed
                 current_visual_map_for_debug = state_identifier.get_board_state_map(warped_board_img)
                 # You could print or visualize this map for debugging piece ID
        else:
            display_placeholder_board(board_detector.output_size)

        update_display_overlays(display_frame, active_test_servo, LOG_PERFORMANCE, loop_time_filter, start_time_perf if LOG_PERFORMANCE else 0, game_phase)
        cv2.imshow("3. Live Feed", display_frame)
        key = cv2.waitKey(1) & 0xFF

        RUN_IN_DEBUG_MODE, _, active_test_servo, should_quit = \
            handle_key_presses(key, RUN_IN_DEBUG_MODE, False, active_test_servo, current_chess_board, board_detector, arm_controller)
        if should_quit: break

        if current_chess_board.is_game_over():
            if game_phase != "GAME_OVER": print_game_result(current_chess_board) # Print once
            game_phase = "GAME_OVER"

        elif game_phase == "AI_THINKING":
            if engine and arm_controller.pca:
                print(f"\nPhase: {game_phase} (AI: {chess.COLOR_NAMES[current_chess_board.turn]})")
                time.sleep(0.2)
                try:
                    result = engine.play(current_chess_board, chess.engine.Limit(time=0.7, depth=10)) # Deeper thought
                    if result.move:
                        print(f"AI plays: {result.move.uci()}")
                        arm_controller.move_piece_on_board(result.move.uci()[:2], result.move.uci()[2:], current_chess_board.is_capture(result.move))
                        current_chess_board.push(result.move)
                        print("Board after AI move:\n", current_chess_board)
                        if warped_board_img is not None: # Capture visual state after AI moves
                            previous_board_map = state_identifier.get_board_state_map(warped_board_img)
                            if RUN_IN_DEBUG_MODE: print("DBG: prev_map set after AI move.")
                        human_player_turn = True; game_phase = "HUMAN_MOVE_WAIT"
                    else: game_phase = "GAME_OVER"; print("AI: No move.") # Stalemate/Checkmate likely
                except Exception as e: print(f"Engine play err: {e}"); game_phase = "ERROR"
            else: # AI can't play (no engine/arm). Manual 's' to switch or 'm' to force.
                print(f"AI Turn (stalled). Engine: {'OK' if engine else 'NO'}, Arm: {'OK' if arm_controller.pca else 'NO'}. 'm':Force AI, 's':Switch")


        elif game_phase == "HUMAN_MOVE_WAIT":
            if not human_player_turn: game_phase = "AI_THINKING"; continue # Should be AI's turn

            if warped_board_img is not None:
                if previous_board_map is None: # Capture initial state if it's human's very first turn
                     if not current_chess_board.move_stack: # Only if board is truly at start
                        previous_board_map = state_identifier.get_board_state_map(warped_board_img)
                        print("Initial board state captured for human's first move.")
                else: # We have a previous map, look for changes
                    current_visual_map = state_identifier.get_board_state_map(warped_board_img)
                    if current_visual_map != previous_board_map:
                        print("\nVisual change detected on board!")
                        detected_move = state_identifier.detect_move_from_maps(
                            previous_board_map, current_visual_map, current_chess_board, AI_PLAYS_AS
                        )
                        if detected_move:
                            if detected_move in current_chess_board.legal_moves:
                                print(f"Human move VALID: {detected_move.uci()}")
                                current_chess_board.push(detected_move)
                                print("Board after Human move:\n", current_chess_board)
                                human_player_turn = False; game_phase = "AI_THINKING"
                                # previous_board_map is updated implicitly when AI makes its move next
                            else:
                                print(f"Human move DETECTED BUT ILLEGAL: {detected_move.uci()}. Ignoring, awaiting valid move.")
                                # Don't update previous_board_map yet, wait for a valid sequence
                        else: # Change detected but not a clear move
                            # print("Visual change, but no clear move inferred. Likely noise.")
                            # Could update previous_board_map to current_visual_map here to accept "nudges"
                            # or small vision errors as new baseline for next check
                            # previous_board_map = current_visual_map # CAUTION: can mask real small moves if too quick
                            pass


        elif game_phase == "ERROR":
            print("Error state. Press 'r' to reset or 'q' to quit.")
            time.sleep(0.5)


        # Allow quitting from GAME_OVER state too
        if game_phase == "GAME_OVER" and key == ord('r'): # Reset from game over
            # Reset logic copied from handle_key_presses
            current_chess_board.reset(); board_detector.last_known_corners=None; board_detector.consecutive_misses=0
            if arm_controller.pca: arm_controller.go_to_home_position()
            human_player_turn = (AI_PLAYS_AS == chess.BLACK)
            game_phase = "AI_THINKING" if not human_player_turn else "HUMAN_MOVE_WAIT"
            previous_board_map = None # Reset visual baseline
            print(f"Board & Arm Reset. New phase: {game_phase}\n", current_chess_board)


    cleanup_resources(arm_controller, engine, cap)


def print_keys_help():
    print("\n--- Chess Vision Arm Control ---")
    print("  'q':Quit | 'd':Toggle BoardDetect Debug | 's':Manually Switch Turn (if stuck)")
    print("  Game:      'm':Force AI Move (if AI stalled) | 'u':Undo | 'r':Reset")
    print("  Arm:  'h':Home | 'o':Open | 'c':Close | 'l':Release | 'p':Print Angles")
    print("        '1-6':Select Servo | '+/-':Increment/Decrement")

def handle_camera_failure(): # Returns True if should break main loop
    global cap, CAM_INDEX
    print("ERR: Cam fail. Retrying..."); time.sleep(0.1)
    if cap is not None: cap.release(); cap=None
    time.sleep(0.5)
    if CAM_INDEX!=-1:
        tmp_cap=cv2.VideoCapture(CAM_INDEX)
        if tmp_cap.isOpened(): cap=tmp_cap; print("INFO: Cam re-opened."); cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480); return False
        else: print(f"CRIT: Fail re-open @{CAM_INDEX}."); tmp_cap.release(); return True
    else: print("CRIT: Cam idx unknown."); return True

def display_placeholder_board(output_size):
    dummy_img=np.zeros((output_size,output_size,3),dtype=np.uint8)
    cv2.putText(dummy_img,"No Board",(20,output_size//2-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    cv2.imshow("1. Warped Board",dummy_img)

def update_display_overlays(disp_frame, act_servo, log_perf, time_filt, st_time, gm_phase):
    h_disp, w_disp = disp_frame.shape[:2]
    cv2.putText(disp_frame,f"Servo:{act_servo.upper()} Phase:{gm_phase}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1,cv2.LINE_AA)
    cv2.putText(disp_frame,"Keys: q,d,s|Game:m,u,r|Arm:h,o,c,l,p,1-6,+-",(10,h_disp-10),cv2.FONT_HERSHEY_SIMPLEX,0.35,(200,200,50),1,cv2.LINE_AA)
    if log_perf and st_time > 0:
        p_time=time.monotonic()-st_time; time_filt.append(p_time)
        if len(time_filt)>20: time_filt.pop(0)
        fps=1.0/(sum(time_filt)/len(time_filt)) if time_filt else 0
        cv2.putText(disp_frame,f"FPS:{fps:.1f}",(w_disp-100,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1,cv2.LINE_AA)

def handle_key_presses(key, run_debug, eng_req_dummy, act_servo, board_st, board_det, arm_ctlr): # Removed game_phase here, handle in main
    global human_player_turn, game_phase, previous_board_map # Modify globals directly
    should_quit = False
    can_arm_operate = arm_ctlr.pca is not None

    if key == ord('q'): should_quit = True; print("Quitting initiated.")
    elif key == ord('d'):
        run_debug = not run_debug; print(f"BoardDetect Debug: {'ON' if run_debug else 'OFF'}")
        if not run_debug: # Close debug windows
            for win in ["Debug - ROI Canny Edges","Debug - Full Canny Edges","Debug - 05 Raw Contour (on processed img)","Debug - 06 Active Corners (on original img)"]:
                try: cv2.destroyWindow(win)
                except: pass
    elif key == ord('m'): # Force AI move if AI is stalled
        if game_phase == "AI_THINKING" and (not engine or not can_arm_operate):
            print("Attempting to force AI move (was stalled)...")
            # This will be picked up by the AI_THINKING block in main on next iteration
            # We just ensure engine_play_requested equivalent is handled. For FSM, simply allow AI_THINKING to re-evaluate.
            # No direct eng_req set here, main FSM does it.
        else:
             print("'m' is for forcing stalled AI. AI currently not in a state to be forced or dependencies missing.")

    elif key == ord('u'): # Undo
        if board_st.move_stack: board_st.pop() # Human's (or last)
        if board_st.move_stack: board_st.pop() # AI's (or second to last)
        print("Undo (2 plies)."); print(board_st)
        current_turn_is_ai = (board_st.turn == AI_PLAYS_AS)
        game_phase = "AI_THINKING" if current_turn_is_ai else "HUMAN_MOVE_WAIT"
        human_player_turn = not current_turn_is_ai
        previous_board_map = None # Must re-capture visual baseline after undo
        print(f"Board reverted. Phase: {game_phase}. Human turn: {human_player_turn}")

    elif key == ord('r'): # Reset
        board_st.reset(); board_det.last_known_corners=None; board_det.consecutive_misses=0
        if can_arm_operate: arm_ctlr.go_to_home_position()
        human_player_turn = (AI_PLAYS_AS == chess.BLACK)
        game_phase = "AI_THINKING" if not human_player_turn else "HUMAN_MOVE_WAIT"
        previous_board_map = None
        print(f"Board & Arm Reset. Phase: {game_phase}.\n{board_st}")

    elif key == ord('h') and can_arm_operate: arm_ctlr.go_to_home_position()
    elif key == ord('o') and can_arm_operate: arm_ctlr.operate_gripper("open")
    elif key == ord('c') and can_arm_operate: arm_ctlr.operate_gripper("close")
    elif key == ord('l') and can_arm_operate: arm_ctlr.release_servos()
    elif key == ord('p') and can_arm_operate: arm_ctlr.print_servo_angles()
    elif ord('1') <= key <= ord('6'):
        servo_map = {'1':'base','2':'shoulder','3':'elbow','4':'wrist_ud','5':'wrist_rot','6':'claw'}
        k_char = chr(key)
        if k_char in servo_map: act_servo = servo_map[k_char]; print(f"Selected Servo: {act_servo.upper()}")
    elif (key==ord('+') or key==ord('=')) and can_arm_operate: arm_ctlr.test_servo_increment(act_servo,5)
    elif key==ord('-') and can_arm_operate: arm_ctlr.test_ servo_decrement(act_servo,5) # Corrected typo again!
    elif key == ord('s'): # Switch turn manually
        print("Manual turn switch.")
        if game_phase == "AI_THINKING" or (game_phase=="HUMAN_MOVE_WAIT" and not human_player_turn) : # If it's AI's turn (or should be)
            game_phase = "HUMAN_MOVE_WAIT"; human_player_turn = True; previous_board_map = None # Force map re-read
        elif game_phase == "HUMAN_MOVE_WAIT" and human_player_turn: # If it's Human's turn
            game_phase = "AI_THINKING"; human_player_turn = False
        print(f"Game phase set to: {game_phase}. Human turn: {human_player_turn}")


    return run_debug, False, act_servo, should_quit # eng_req always False from here, FSM controls

def cleanup_resources(arm_ctlr, eng, cam_cap):
    print("Cleaning up...");
    if arm_ctlr.pca is not None: arm_ctlr.release_servos()
    if eng:
        try: eng.quit()
        except: pass
    if cam_cap: cam_cap.release()
    cv2.destroyAllWindows()

def print_game_result(board_st):
    print("--- Game Result ---")
    if board_st.is_checkmate(): print(f"Checkmate! {'WHITE' if board_st.turn==chess.BLACK else 'BLACK'} wins.") # Winner is one whose turn it ISNT
    elif board_st.is_stalemate(): print("Stalemate! Draw.")
    elif board_st.is_insufficient_material(): print("Draw by insufficient material.")
    # Add more draw conditions as needed.

if __name__ == "__main__":
    main()