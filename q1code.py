import cv2
import numpy as np
import chess
import chess.engine
import time
import os

# --- PCA9685 & Servo Control Imports ---
import board # For board.SCL, board.SDA
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo as adafruit_servo_motor # To avoid confusion with a potential custom servo class

# (BoardDetector class remains the same as your previous "stable" version)
# For brevity, I will omit it here but assume it's present in your file.
# Ensure the BoardDetector class is defined before ArmController or main()
# --- START OF BoardDetector CLASS (PASTE PREVIOUS STABLE VERSION HERE) ---
class BoardDetector:
    def __init__(self, output_size=480):
        self.output_size = output_size
        self.last_known_corners = None
        self.consecutive_misses = 0
        self.blur_kernel_size = (5, 5)
        self.canny_threshold1 = 40
        self.canny_threshold2 = 120
        self.approx_poly_epsilon_factor = 0.02
        self.min_area_ratio = 0.02
        self.max_area_ratio = 0.90
        self.corner_smoothing_alpha = 0.6
        self.max_consecutive_misses_before_reset = 7
        self.roi_search_expansion_factor = 1.3
        self.max_corner_drift_for_smoothing = 70

    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        remaining_pts = []
        for pt in pts:
            if not np.array_equal(pt, rect[0]) and not np.array_equal(pt, rect[2]):
                remaining_pts.append(pt)
        if len(remaining_pts) == 2:
            pt_A, pt_B = remaining_pts[0], remaining_pts[1]
            diff_A = pt_A[1] - pt_A[0] 
            diff_B = pt_B[1] - pt_B[0]
            if diff_A < diff_B: rect[1], rect[3] = pt_A, pt_B
            else: rect[1], rect[3] = pt_B, pt_A
        elif pts.shape[0] !=4 : return None
        return rect

    def _find_board_contour_in_image(self, image_to_search, image_area_for_ratios, is_roi_search, debug_mode=False, original_full_image_for_debug=None):
        gray = cv2.cvtColor(image_to_search, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel_size, 0)
        current_canny_t1, current_canny_t2 = self.canny_threshold1, self.canny_threshold2
        if is_roi_search:
            current_canny_t1 = max(10, self.canny_threshold1 - 15)
            current_canny_t2 = max(30, self.canny_threshold2 - 30)
        edged = cv2.Canny(blurred, current_canny_t1, current_canny_t2)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        for c in contours:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, self.approx_poly_epsilon_factor * perimeter, True)
            if len(approx) == 4:
                contour_area = cv2.contourArea(approx)
                is_convex = cv2.isContourConvex(approx)
                min_abs_area = self.min_area_ratio * image_area_for_ratios
                max_abs_area = self.max_area_ratio * image_area_for_ratios
                if min_abs_area < contour_area < max_abs_area and is_convex:
                    if debug_mode and not is_roi_search and original_full_image_for_debug is not None:
                        selected_board_img = original_full_image_for_debug.copy()
                        cv2.drawContours(selected_board_img, [approx.reshape(4,2)], -1, (0,255,0), 3)
                        cv2.imshow("Debug - 05 Selected Raw Contour", selected_board_img)
                    return approx.reshape(4, 2).astype(np.float32)
        return None

    def detect(self, image, debug_mode=False):
        img_h, img_w = image.shape[:2]; full_image_area = img_h * img_w
        current_raw_corners = None; roi_used_for_detection = False
        if self.last_known_corners is not None and self.consecutive_misses < (self.max_consecutive_misses_before_reset // 2):
            x_coords=self.last_known_corners[:,0]; y_coords=self.last_known_corners[:,1]
            min_x,max_x=np.min(x_coords),np.max(x_coords); min_y,max_y=np.min(y_coords),np.max(y_coords)
            center_x,center_y=(min_x+max_x)/2,(min_y+max_y)/2
            roi_width=int((max_x-min_x)*self.roi_search_expansion_factor); roi_height=int((max_y-min_y)*self.roi_search_expansion_factor)
            roi_x1=max(0,int(center_x-roi_width/2)); roi_y1=max(0,int(center_y-roi_height/2))
            roi_x2=min(img_w,roi_x1+roi_width); roi_y2=min(img_h,roi_y1+roi_height)
            if(roi_x2-roi_x1)>20 and (roi_y2-roi_y1)>20:
                image_roi=image[roi_y1:roi_y2,roi_x1:roi_x2]
                if image_roi.size>0:
                    current_raw_corners_roi=self._find_board_contour_in_image(image_roi,image_roi.shape[0]*image_roi.shape[1],True,debug_mode)
                    if current_raw_corners_roi is not None:
                        current_raw_corners=current_raw_corners_roi+np.array([roi_x1,roi_y1],dtype=np.float32)
                        roi_used_for_detection=True
                        if debug_mode: print("DEBUG: Board found in ROI.")
        if current_raw_corners is None:
            roi_used_for_detection=False
            if debug_mode and self.last_known_corners is not None and self.consecutive_misses < (self.max_consecutive_misses_before_reset // 2): print("DEBUG: Not found in ROI, trying full image search.")
            current_raw_corners = self._find_board_contour_in_image(image, full_image_area, False, debug_mode, original_full_image_for_debug=image)
        processed_corners_for_warp = None
        if current_raw_corners is not None:
            ordered_raw_corners = self._order_points(current_raw_corners)
            if ordered_raw_corners is None: current_raw_corners = None
            else:
                if self.last_known_corners is not None:
                    avg_drift=np.mean(np.linalg.norm(ordered_raw_corners-self.last_known_corners,axis=1))
                    if avg_drift < self.max_corner_drift_for_smoothing: processed_corners_for_warp=(self.corner_smoothing_alpha*ordered_raw_corners+(1-self.corner_smoothing_alpha)*self.last_known_corners)
                    else: processed_corners_for_warp=ordered_raw_corners; print(f"DEBUG: Large drift ({avg_drift:.1f}px), using raw.")
                else: processed_corners_for_warp=ordered_raw_corners
                self.last_known_corners=processed_corners_for_warp.copy(); self.consecutive_misses=0
        else:
            self.consecutive_misses+=1
            if self.last_known_corners is not None and self.consecutive_misses<self.max_consecutive_misses_before_reset:
                processed_corners_for_warp=self.last_known_corners
                if debug_mode:print(f"DEBUG: Coasting (miss #{self.consecutive_misses}).")
            else:
                if debug_mode and self.last_known_corners is not None:print("DEBUG: Max misses reached, resetting.")
                self.last_known_corners=None; processed_corners_for_warp=None
        if processed_corners_for_warp is None: return None,None
        dst_pts=np.array([[0,0],[self.output_size-1,0],[self.output_size-1,self.output_size-1],[0,self.output_size-1]],dtype="float32")
        try:
            M=cv2.getPerspectiveTransform(processed_corners_for_warp,dst_pts)
            warped_board=cv2.warpPerspective(image,M,(self.output_size,self.output_size))
        except cv2.error as e:
            if debug_mode:print(f"DEBUG: Warp Error (bad corners: {processed_corners_for_warp}): {e}")
            return None,processed_corners_for_warp
        if debug_mode:
            active_corners_img=image.copy()
            if roi_used_for_detection and 'roi_x1' in locals(): cv2.rectangle(active_corners_img,(roi_x1,roi_y1),(roi_x2,roi_y2),(255,165,0),2)
            for i,p in enumerate(processed_corners_for_warp): cv2.circle(active_corners_img,tuple(np.int0(p)),5,[(255,0,0),(0,255,0),(0,0,255),(255,255,0)][i%4],-1); cv2.putText(active_corners_img,str(i),tuple(np.int0(p)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
            cv2.imshow("Debug - 06 Active Corners",active_corners_img)
        return warped_board, processed_corners_for_warp
# --- END OF BoardDetector CLASS ---


# --- Arm Controller Class ---
class ArmController:
    # YOU MUST UPDATE THESE CHANNEL ASSIGNMENTS TO MATCH YOUR WIRING
    SERVO_CHANNELS = {
        'base': 0,       # MG995
        'shoulder': 1,   # MG995
        'elbow': 2,      # MG995
        'wrist_ud': 3,   # MG90S (up/down)
        'wrist_rot': 4,  # MG90S (sideways rotation)
        'claw': 5,       # MG90S
    }

    # SERVO CALIBRATION: (min_pulse, max_pulse, default_angle, min_angle, max_angle)
    # ** THESE ARE GENERIC DEFAULTS - YOU MUST CALIBRATE EACH SERVO! **
    # min_pulse and max_pulse are in microseconds (us).
    # Typical values: MG995/MG90S often 500-2500us for 0-180 deg.
    # Test carefully to find safe operational range for each servo.
    SERVO_PARAMS = {
        'base':      {'min_pulse': 500, 'max_pulse': 2500, 'default_angle': 90,  'min_angle': 0, 'max_angle': 180, 'current_angle': 90}, # MG995
        'shoulder':  {'min_pulse': 500, 'max_pulse': 2500, 'default_angle': 135, 'min_angle': 45, 'max_angle': 180, 'current_angle': 135},# MG995
        'elbow':     {'min_pulse': 500, 'max_pulse': 2500, 'default_angle': 90, 'min_angle': 0, 'max_angle': 180, 'current_angle': 90}, # MG995
        'wrist_ud':  {'min_pulse': 500, 'max_pulse': 2500, 'default_angle': 90, 'min_angle': 0, 'max_angle': 180, 'current_angle': 90},  # MG90S
        'wrist_rot': {'min_pulse': 500, 'max_pulse': 2500, 'default_angle': 90, 'min_angle': 0, 'max_angle': 180, 'current_angle': 90},# MG90S
        'claw':      {'min_pulse': 500, 'max_pulse': 2500, 'default_angle': 90, 'min_angle': 30, 'max_angle': 120, 'current_angle': 90} # MG90S (Claw: smaller range often used)
    }
    # For claw, 30 might be open, 120 might be closed, or vice versa. Calibrate!


    def __init__(self):
        self.servos = {}
        try:
            # Initialize I2C bus
            i2c = busio.I2C(board.SCL, board.SDA)
            # Initialize PCA9685
            self.pca = PCA9685(i2c)
            self.pca.frequency = 50  # Set PWM frequency to 50Hz (standard for most servos)
            print("PCA9685 initialized successfully.")

            # Initialize servo objects
            for name, channel in self.SERVO_CHANNELS.items():
                params = self.SERVO_PARAMS[name]
                # The adafruit_motor.servo.Servo class takes PCA9685 channel object
                servo_obj = adafruit_servo_motor.Servo(
                    self.pca.channels[channel],
                    min_pulse=params['min_pulse'],
                    max_pulse=params['max_pulse'],
                    actuation_range=params['max_angle'] - params['min_angle'] # Full range
                )
                self.servos[name] = servo_obj
                # Set initial default angle
                self.set_servo_angle(name, params['default_angle'], initial_setup=True)
                time.sleep(0.1) # Small delay between servo initializations
            print("All servos initialized to default positions.")

        except ValueError as e:
            print(f"Error: Could not find PCA9685. Check I2C connections and address. {e}")
            self.pca = None
        except Exception as e:
            print(f"An unexpected error occurred during ArmController initialization: {e}")
            self.pca = None
            
    def set_servo_angle(self, servo_name, angle, initial_setup=False):
        if not self.pca or servo_name not in self.servos:
            if not initial_setup: # Don't print error during initial mass setup if pca failed
                 print(f"Error: PCA not initialized or servo '{servo_name}' not found.")
            return

        servo_obj = self.servos[servo_name]
        params = self.SERVO_PARAMS[servo_name]
        
        # Clamp angle to servo's defined min/max
        clamped_angle = max(params['min_angle'], min(params['max_angle'], angle))

        # The adafruit_motor.servo.Servo.angle property maps its 0-actuation_range
        # input to the min_pulse/max_pulse range.
        # So, if min_angle is not 0, we need to offset.
        # E.g., if min_angle=30, max_angle=150 (range=120), then angle=30 becomes 0 for servo_obj.angle
        angle_for_servo_lib = clamped_angle - params['min_angle']
        
        try:
            servo_obj.angle = angle_for_servo_lib
            self.SERVO_PARAMS[servo_name]['current_angle'] = clamped_angle # Store the actual angle
            if not initial_setup:
                print(f"Servo '{servo_name}' set to {clamped_angle:.1f} degrees (library angle: {angle_for_servo_lib:.1f}).")
        except Exception as e:
            print(f"Error setting angle for servo '{servo_name}': {e}")

    def go_to_home_position(self):
        print("Arm moving to home position...")
        for name, params in self.SERVO_PARAMS.items():
            self.set_servo_angle(name, params['default_angle'])
            time.sleep(0.2) # Move servos one by one or with small delay
        print("Arm at home position.")

    def operate_gripper(self, state): # state can be "open" or "close"
        # ** YOU NEED TO CALIBRATE THESE ANGLES **
        open_angle = self.SERVO_PARAMS['claw']['min_angle']  # Example: smaller angle = open
        closed_angle = self.SERVO_PARAMS['claw']['max_angle'] # Example: larger angle = closed

        if state == "open":
            self.set_servo_angle('claw', open_angle)
            print("Gripper: Opening")
        elif state == "close":
            self.set_servo_angle('claw', closed_angle)
            print("Gripper: Closing")
        else:
            print(f"Gripper: Unknown state '{state}'. Use 'open' or 'close'.")

    def test_servo_increment(self, servo_name, increment=5):
        """Incrementally moves a servo for testing/calibration."""
        if servo_name not in self.SERVO_PARAMS:
            print(f"Unknown servo: {servo_name}")
            return
        current_angle = self.SERVO_PARAMS[servo_name]['current_angle']
        new_angle = current_angle + increment
        self.set_servo_angle(servo_name, new_angle)
        
    def test_servo_decrement(self, servo_name, decrement=5):
        """Decrementally moves a servo for testing/calibration."""
        if servo_name not in self.SERVO_PARAMS:
            print(f"Unknown servo: {servo_name}")
            return
        current_angle = self.SERVO_PARAMS[servo_name]['current_angle']
        new_angle = current_angle - decrement
        self.set_servo_angle(servo_name, new_angle)

    def move_piece_on_board(self, from_square_uci, to_square_uci, captured_piece=False):
        """
        High-level function to move a chess piece.
        THIS IS A COMPLEX PLACEHOLDER - Requires kinematics, path planning,
        and board-to-arm coordinate transformation.
        """
        print(f"ARM ACTION (Placeholder): Pick from {from_square_uci}, Place at {to_square_uci}")
        
        # 1. Transform from_square_uci to arm coordinates (X, Y, Z_hover)
        #    (This needs calibration: board origin relative to arm, square size)
        #    coord_from = self.board_square_to_arm_xyz(from_square_uci)
        #    coord_to = self.board_square_to_arm_xyz(to_square_uci)

        # 2. Calculate joint angles using Inverse Kinematics (IK) for coord_from
        #    angles_from_hover = self.inverse_kinematics(coord_from['x'], coord_from['y'], Z_HOVER)
        
        # 3. Sequence:
        #    self.move_joints_to(angles_from_hover) # Move above 'from' square
        #    self.operate_gripper("open")
        #    self.move_joints_to(angles_from_pick)  # Lower to pick (Z_PICK)
        #    self.operate_gripper("close")
        #    time.sleep(0.5) # Ensure grip
        #    self.move_joints_to(angles_from_hover) # Lift piece
        
        #    angles_to_hover = self.inverse_kinematics(coord_to['x'], coord_to['y'], Z_HOVER)
        #    self.move_joints_to(angles_to_hover)   # Move above 'to' square
        #    self.move_joints_to(angles_to_place)   # Lower to place (Z_PLACE)
        #    self.operate_gripper("open")
        #    time.sleep(0.5) # Ensure release
        #    self.move_joints_to(angles_to_hover)   # Lift arm
        
        #    self.go_to_home_position()
        
        # If captured_piece, need a routine to move it to a 'captured pieces' zone.
        
        print("ARM ACTION (Placeholder): Movement sequence would occur here.")
        time.sleep(2) # Simulate arm movement time

    def release_servos(self):
        """ Frees all servos by setting their duty cycle to 0. Useful to prevent buzzing. """
        if self.pca:
            for name in self.servos:
                try:
                    self.pca.channels[self.SERVO_CHANNELS[name]].duty_cycle = 0
                    print(f"Servo '{name}' released.")
                except Exception as e:
                    print(f"Error releasing servo {name}: {e}")
        else:
            print("PCA not initialized, cannot release servos.")


# --- Initialize Webcam (Keep existing robust version) ---
CAM_PIPELINE_USB = 0 
cap = None
# ... (Paste your working camera initialization code here) ...
try:
    cap = cv2.VideoCapture(CAM_PIPELINE_USB) 
    if not cap.isOpened():
        print(f"INFO: Webcam at index {CAM_PIPELINE_USB} failed, trying index 1...")
        CAM_PIPELINE_USB = 1
        cap = cv2.VideoCapture(CAM_PIPELINE_USB)
        if not cap.isOpened():
             raise IOError(f"Cannot open webcam. Tried indices 0 and 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"Webcam opened successfully (index {CAM_PIPELINE_USB}, W: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, H: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}).")
except Exception as e:
    print(f"Error initializing webcam: {e}")
    exit()

# --- Load the chess engine (Keep existing version) ---
engine_path = "/usr/games/stockfish"
engine = None
# ... (Paste your working engine loading code here) ...
try:
    if not os.path.exists(engine_path) or not os.access(engine_path, os.X_OK):
        raise FileNotFoundError(f"Stockfish not found or not executable at {engine_path}")
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    print(f"Stockfish engine loaded successfully from: {engine_path}")
except Exception as e:
    print(f"Error: Could not load Stockfish engine from '{engine_path}'. Error: {e}")
    engine = None


# --- Chess game state & Utility Functions (Keep existing) ---
current_chess_board = chess.Board()
# ... (segment_board, highlight_pawn_positions, get_user_move - unchanged) ...
def segment_board(img): # Unchanged
    if img is None or img.shape[0] == 0 or img.shape[1] == 0: return []
    height, width = img.shape[:2]; square_h, square_w = height // 8, width // 8; squares = []
    for row in range(8):
        for col in range(8): y1,y2=row*square_h,(row+1)*square_h; x1,x2=col*square_w,(col+1)*square_w; square_img=img[y1:y2,x1:x2]; squares.append(((row,col),square_img))
    return squares

def highlight_pawn_positions(img, b): # Unchanged
    if img is None or img.shape[0]==0: return img
    sh,sw=img.shape[0]//8,img.shape[1]//8; hi=img.copy()
    for sq_idx in chess.SQUARES:
        p=b.piece_at(sq_idx)
        if p and p.piece_type==chess.PAWN:
            r,c=7-chess.square_rank(sq_idx),chess.square_file(sq_idx)
            tl,br=(c*sw,r*sh),((c+1)*sw,(r+1)*sh)
            clr=(0,255,0) if p.color==chess.WHITE else (0,0,255)
            cv2.rectangle(hi,tl,br,clr,2)
    return hi

def get_user_move(prev_img, new_img): return None # Unchanged


# --- Main Loop ---
def main():
    global current_chess_board
    warped_board_output_size = 480 
    board_detector = BoardDetector(output_size=warped_board_output_size)
    arm_controller = ArmController() # Initialize the arm

    RUN_IN_DEBUG_MODE = False
    print("\n--- Starting Chess Vision with Robotic Arm ---")
    print("Controls: 'm'=Engine Move, 'u'=Undo, 'r'=Reset, 'q'=Quit")
    print("Arm Test: 'h'=Arm Home, 'o'=Open Claw, 'c'=Close Claw")
    print("Arm Calibrate: '1-6' to select servo, '+'/'-' to move, 's'=Save angles (TBD), 'l'=Release Servos")
    
    active_test_servo = 'base' # For calibration key presses

    if arm_controller.pca is None:
        print("WARNING: Arm controller not initialized due to PCA9685 error. Arm functions disabled.")

    engine_move_requested = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame. Retrying..."); time.sleep(0.1)
            ret, frame = cap.read();
            if not ret: print("Error: Still failed. Exiting."); break
        
        processed_frame = frame 
        warped_board_img, board_corners_in_original = board_detector.detect(
            processed_frame.copy(), debug_mode=RUN_IN_DEBUG_MODE
        )
        display_frame = processed_frame.copy()

        if warped_board_img is not None:
            cv2.imshow("1. Warped Chessboard", warped_board_img)
            highlighted_warped_board = highlight_pawn_positions(warped_board_img, current_chess_board)
            cv2.imshow("2. Warped + Pawns", highlighted_warped_board)
            if board_corners_in_original is not None:
                cv2.polylines(display_frame, [np.int32(board_corners_in_original)], True, (0,255,0),2,cv2.LINE_AA)
        else:
            dummy = np.zeros((warped_board_output_size,warped_board_output_size,3),dtype=np.uint8)
            cv2.putText(dummy,"Board Not Detected",(20,warped_board_output_size//2-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            cv2.imshow("1. Warped Chessboard",dummy); cv2.imshow("2. Warped + Pawns",dummy)

        status_text = f"Test Servo: {active_test_servo.upper()}"
        cv2.putText(display_frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(display_frame, "m:eng u:undo r:reset q:quit | h:home o:open c:close | 1-6 sel servo, +/- move", (10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,50),1,cv2.LINE_AA)
        cv2.imshow("3. Live Feed with Detection", display_frame)

        key = cv2.waitKey(1) & 0xFF

        # --- Engine Move Logic ---
        if engine_move_requested and engine and not current_chess_board.is_game_over() and arm_controller.pca:
            engine_move_requested = False # Reset flag
            try:
                print("Engine thinking for physical move...")
                result = engine.play(current_chess_board, chess.engine.Limit(time=0.2, depth=5)) 
                if result.move:
                    from_sq_uci = result.move.uci()[:2]
                    to_sq_uci = result.move.uci()[2:]
                    print(f"Engine plays: {result.move.uci()}")
                    # Physical arm movement
                    # TODO: Check if to_sq has a piece for capture logic
                    is_capture = current_chess_board.is_capture(result.move)
                    arm_controller.move_piece_on_board(from_sq_uci, to_sq_uci, captured_piece=is_capture)
                    
                    current_chess_board.push(result.move) # Update internal board AFTER physical move
                    print(current_chess_board)
                else:
                    print("Engine could not find a valid move.")
            except Exception as e:
                print(f"Engine or Arm error during play: {e}")

        # --- Key Press Handling ---
        if key == ord('m'):
            if engine and not current_chess_board.is_game_over() and arm_controller.pca:
                engine_move_requested = True
                print("Engine move requested, will process soon.")
            elif not arm_controller.pca: print("Arm not ready for engine move.")
            elif not engine: print("Engine not loaded.")
            elif current_chess_board.is_game_over(): print("Game is over.")

        elif key == ord('u'): # Undo
            if current_chess_board.move_stack:
                current_chess_board.pop();
                if current_chess_board.move_stack: current_chess_board.pop() # Pop engine and human if applicable
                print("Last one/two moves undone."); print(current_chess_board)
            else: print("No moves to undo.")
        elif key == ord('r'): # Reset
            current_chess_board.reset(); board_detector.last_known_corners=None; board_detector.consecutive_misses=0
            if arm_controller.pca: arm_controller.go_to_home_position()
            print("Board reset."); print(current_chess_board)
        
        # Arm Control Keys
        elif key == ord('h') and arm_controller.pca: arm_controller.go_to_home_position()
        elif key == ord('o') and arm_controller.pca: arm_controller.operate_gripper("open")
        elif key == ord('c') and arm_controller.pca: arm_controller.operate_gripper("close")
        elif key == ord('l') and arm_controller.pca: arm_controller.release_servos()

        # Servo Selection for Testing/Calibration
        elif key == ord('1'): active_test_servo = 'base'; print("Selected: base")
        elif key == ord('2'): active_test_servo = 'shoulder'; print("Selected: shoulder")
        elif key == ord('3'): active_test_servo = 'elbow'; print("Selected: elbow")
        elif key == ord('4'): active_test_servo = 'wrist_ud'; print("Selected: wrist_ud")
        elif key == ord('5'): active_test_servo = 'wrist_rot'; print("Selected: wrist_rot")
        elif key == ord('6'): active_test_servo = 'claw'; print("Selected: claw")
        
        # Servo Movement for Testing/Calibration
        elif key == ord('+') or key == ord('='): # + key
            if arm_controller.pca: arm_controller.test_servo_increment(active_test_servo, 5)
        elif key == ord('-'): # - key
            if arm_controller.pca: arm_controller.test_servo_decrement(active_test_servo, 5)
            
        elif key == ord('q'): print("Quitting."); break
        
        if current_chess_board.is_game_over() and not RUN_IN_DEBUG_MODE:
            print("Game Over."); time.sleep(2); break 
    
    print("Exiting program...")
    if arm_controller.pca: arm_controller.release_servos() # Release servos on exit
    if engine:
        try: engine.quit()
        except: pass
    if cap: cap.release()
    cv2.destroyAllWindows()

    if current_chess_board.is_checkmate(): print(f"Checkmate! Winner: {'Black' if current_chess_board.turn == chess.WHITE else 'White'}")
    # ... other game end messages
if __name__ == "__main__":
    main()