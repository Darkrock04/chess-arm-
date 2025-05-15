import cv2
import numpy as np
import chess
import chess.engine
import time
import os

# --- PCA9685 & Servo Control Imports ---
# Ensure these are installed: sudo pip3 install adafruit-circuitpython-pca9685 adafruit-circuitpython-motor
import board 
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo as adafruit_servo_motor

# --- BoardDetector CLASS (Paste your stable version from previous response here) ---
# For brevity, assuming it's correctly defined. If not, let me know.
class BoardDetector:
    def __init__(self, output_size=480):
        self.output_size = output_size; self.last_known_corners = None; self.consecutive_misses = 0
        self.blur_kernel_size = (5,5); self.canny_threshold1 = 40; self.canny_threshold2 = 120
        self.approx_poly_epsilon_factor = 0.02; self.min_area_ratio = 0.02; self.max_area_ratio = 0.90
        self.corner_smoothing_alpha = 0.6; self.max_consecutive_misses_before_reset = 7
        self.roi_search_expansion_factor = 1.3; self.max_corner_drift_for_smoothing = 70
    def _order_points(self,pts):
        rect=np.zeros((4,2),dtype="float32");s=pts.sum(axis=1);rect[0]=pts[np.argmin(s)];rect[2]=pts[np.argmax(s)]
        rp=[];[rp.append(pt)for pt in pts if not np.array_equal(pt,rect[0])and not np.array_equal(pt,rect[2])]
        if len(rp)==2:pa,pb=rp[0],rp[1];da=pa[1]-pa[0];db=pb[1]-pb[0];rect[1],rect[3]=(pa,pb)if da<db else(pb,pa)
        elif pts.shape[0]!=4:return None
        return rect
    def _find_board_contour_in_image(self,img_search,img_area_ratios,is_roi,dbg=False,orig_img_dbg=None):
        gray=cv2.cvtColor(img_search,cv2.COLOR_BGR2GRAY);blr=cv2.GaussianBlur(gray,self.blur_kernel_size,0)
        ct1,ct2=self.canny_threshold1,self.canny_threshold2
        if is_roi:ct1,ct2=max(10,ct1-15),max(30,ct2-30)
        edg=cv2.Canny(blr,ct1,ct2);cnts,_=cv2.findContours(edg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:return None
        cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:5]
        for c in cnts:
            p=cv2.arcLength(c,True);ap=cv2.approxPolyDP(c,self.approx_poly_epsilon_factor*p,True)
            if len(ap)==4:
                ca=cv2.contourArea(ap);ic=cv2.isContourConvex(ap)
                min_a,max_a=self.min_area_ratio*img_area_ratios,self.max_area_ratio*img_area_ratios
                if min_a<ca<max_a and ic:
                    if dbg and not is_roi and orig_img_dbg is not None:
                        sbi=orig_img_dbg.copy();cv2.drawContours(sbi,[ap.reshape(4,2)],-1,(0,255,0),3);cv2.imshow("Debug - 05 Raw Contour",sbi)
                    return ap.reshape(4,2).astype(np.float32)
        return None
    def detect(self,image,debug_mode=False):
        ih,iw=image.shape[:2];fia=ih*iw;crc=None;roi_used=False
        if self.last_known_corners is not None and self.consecutive_misses<(self.max_consecutive_misses_before_reset//2):
            xc,yc=self.last_known_corners[:,0],self.last_known_corners[:,1];mnx,mxx=np.min(xc),np.max(xc);mny,mxy=np.min(yc),np.max(yc)
            cx,cy=(mnx+mxx)/2,(mny+mxy)/2;rw,rh=int((mxx-mnx)*self.roi_search_expansion_factor),int((mxy-mny)*self.roi_search_expansion_factor)
            rx1,ry1=max(0,int(cx-rw/2)),max(0,int(cy-rh/2));rx2,ry2=min(iw,rx1+rw),min(ih,ry1+rh)
            if(rx2-rx1)>20 and(ry2-ry1)>20:
                roi=image[ry1:ry2,rx1:rx2]
                if roi.size>0:
                    crc_roi=self._find_board_contour_in_image(roi,roi.shape[0]*roi.shape[1],True,debug_mode)
                    if crc_roi is not None:crc=crc_roi+np.array([rx1,ry1],dtype=np.float32);roi_used=True; #if debug_mode:print("DBG: ROI Hit")
        if crc is None:
            roi_used=False #if debug_mode and self.last_known_corners is not None and self.consecutive_misses < (self.max_consecutive_misses_before_reset//2): print("DBG: ROI Miss, Full Search")
            crc=self._find_board_contour_in_image(image,fia,False,debug_mode,original_full_image_for_debug=image)
        pcfw=None
        if crc is not None:
            orc=self._order_points(crc)
            if orc is None:crc=None
            else:
                if self.last_known_corners is not None:
                    ad=np.mean(np.linalg.norm(orc-self.last_known_corners,axis=1))
                    if ad<self.max_corner_drift_for_smoothing:pcfw=(self.corner_smoothing_alpha*orc+(1-self.corner_smoothing_alpha)*self.last_known_corners)
                    else:pcfw=orc #if debug_mode:print(f"DBG: Drift {ad:.1f}px, raw.")
                else:pcfw=orc
                self.last_known_corners=pcfw.copy();self.consecutive_misses=0
        else:
            self.consecutive_misses+=1
            if self.last_known_corners is not None and self.consecutive_misses<self.max_consecutive_misses_before_reset:
                pcfw=self.last_known_corners #if debug_mode:print(f"DBG: Coast miss #{self.consecutive_misses}")
            else: #if debug_mode and self.last_known_corners is not None:print("DBG: Max miss, reset.")
                self.last_known_corners=None;pcfw=None
        if pcfw is None:return None,None
        dst=np.array([[0,0],[self.output_size-1,0],[self.output_size-1,self.output_size-1],[0,self.output_size-1]],dtype="float32")
        try:M=cv2.getPerspectiveTransform(pcfw,dst);wb=cv2.warpPerspective(image,M,(self.output_size,self.output_size))
        except cv2.error: #if debug_mode:print(f"DBG: Warp Err {pcfw}")
            return None,pcfw
        if debug_mode:
            aci=image.copy()
            if roi_used and'rx1'in locals():cv2.rectangle(aci,(rx1,ry1),(rx2,ry2),(255,165,0),2)
            for i,p in enumerate(pcfw):cv2.circle(aci,tuple(np.int0(p)),5,[(255,0,0),(0,255,0),(0,0,255),(255,255,0)][i%4],-1);cv2.putText(aci,str(i),tuple(np.int0(p)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
            cv2.imshow("Debug - 06 Active Corners",aci)
        return wb,pcfw

# --- Arm Controller Class ---
class ArmController:
    # !!! CRITICAL: UPDATE THESE CHANNEL ASSIGNMENTS TO MATCH YOUR WIRING ON PCA9685 (0-15) !!!
    SERVO_CHANNELS = {
        'base': 0,        # MG995
        'shoulder': 1,    # MG995
        'elbow': 2,       # MG995
        'wrist_ud': 3,    # MG90S (up/down)
        'wrist_rot': 4,   # MG90S (sideways rotation / roll)
        'claw': 5,        # MG90S
    }

    # !!! CRITICAL: MANUALLY CALIBRATE EACH SERVO'S PARAMETERS !!!
    # These are GENERIC starting points. Incorrect values WILL cause problems.
    # Use keys '1'-'6' to select servo, then '+' / '-' to move carefully and find limits.
    # 'min_angle' & 'max_angle': Your desired operational range for that joint in degrees.
    # 'default_angle': The angle for the "home" position.
    # 'min_pulse' & 'max_pulse': Microsecond pulse widths corresponding to the Adafruit library's
    #                            internal 0 to (max_angle - min_angle) degree mapping for THIS servo.
    #                            Common hobby servos ~500us to ~2500us often map to ~180 degrees of rotation.
    #                            This means `actuation_range` for adafruit_motor.servo is usually 180.
    #                            However, the critical part is finding the pulse values that give YOUR servo its
    #                            full, safe mechanical range mapped to YOUR `min_angle` and `max_angle`.
    # 'current_angle': Internal tracking.
    SERVO_PARAMS = {
        #                   min_p, max_p, def_ang, min_ang, max_ang, cur_ang
        'base':      {'min_pulse': 500, 'max_pulse': 2500, 'default_angle': 90,  'min_angle': 0,   'max_angle': 180, 'current_angle': 90},
        'shoulder':  {'min_pulse': 500, 'max_pulse': 2500, 'default_angle': 135, 'min_angle': 30,  'max_angle': 150, 'current_angle': 135},# e.g., raised
        'elbow':     {'min_pulse': 500, 'max_pulse': 2500, 'default_angle': 45,  'min_angle': 0,   'max_angle': 135, 'current_angle': 45}, # e.g., bent forward
        'wrist_ud':  {'min_pulse': 500, 'max_pulse': 2500, 'default_angle': 90,  'min_angle': 0,   'max_angle': 180, 'current_angle': 90},
        'wrist_rot': {'min_pulse': 500, 'max_pulse': 2500, 'default_angle': 90,  'min_angle': 0,   'max_angle': 180, 'current_angle': 90},
        'claw':      {'min_pulse': 600, 'max_pulse': 2400, 'default_angle': 45,  'min_angle': 45,  'max_angle': 110, 'current_angle': 45} # Example: 45=open, 110=closed
    }
    # NOTE: For `adafruit_motor.servo.Servo`, the `actuation_range` parameter
    # should generally correspond to the physical range the `min_pulse` and `max_pulse` give.
    # Typically 180 for standard servos, but can be less.
    # Our `min_angle` and `max_angle` define the *useful portion* of that range.

    def __init__(self):
        self.servos = {}
        self.pca = None
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
                
                # Calculate actuation_range based on your min_angle and max_angle
                # The library maps 0 to `actuation_range_for_lib` to `min_pulse` to `max_pulse`.
                # We will control it from `params['min_angle']` to `params['max_angle']`.
                actuation_range_for_lib = 180 # Assume standard 180deg range for min/max pulse. Adjust if your servo is e.g. 270deg for these pulses.
                                             # Some prefer to set this to `params['max_angle'] - params['min_angle']` if min/max pulse map directly.
                                             # Let's stick to a common 180 for pulse range, and our code maps our desired angles.

                servo_obj = adafruit_servo_motor.Servo(
                    self.pca.channels[channel_num],
                    min_pulse=params['min_pulse'],
                    max_pulse=params['max_pulse'],
                    actuation_range = actuation_range_for_lib # Total range servo supports for given pulses
                )
                self.servos[name] = servo_obj
                params['current_angle'] = params['default_angle'] # Ensure current_angle matches default
                self.set_servo_angle(name, params['default_angle'], initial_setup=True)
                time.sleep(0.05) 
            print("Servos initialized (attempted). VERIFY POSITIONS AND CALIBRATE!")

        except ValueError as e: # PCA9685 not found
            print(f"CRITICAL Error: Could not find PCA9685. Check I2C wiring & sudo i2cdetect -y 1. Error: {e}")
        except Exception as e:
            print(f"CRITICAL Error during ArmController init: {e}")
            
    def set_servo_angle(self, servo_name, angle, initial_setup=False):
        if not self.pca or servo_name not in self.servos:
            if not initial_setup and self.pca: print(f"Warn: Servo '{servo_name}' not in self.servos dict.")
            return

        servo_obj = self.servos[servo_name]
        params = self.SERVO_PARAMS[servo_name]
        
        clamped_angle = max(params['min_angle'], min(params['max_angle'], float(angle)))

        # The adafruit_motor.servo.Servo.angle property maps its 0 -> `actuation_range`
        # input to the physical servo's range defined by min_pulse and max_pulse.
        # We need to map our `clamped_angle` (which is in our defined min_angle to max_angle)
        # to the 0 -> `actuation_range` that the library expects.
        # Example: if actuation_range=180, min_pulse=500, max_pulse=2500.
        # If our useful servo range is min_angle=30, max_angle=150:
        # A desired angle of 30 (clamped_angle) should map to 30 for the library IF actuation_range=180 was set.
        # A desired angle of 150 should map to 150 for the library.
        # This means direct angle setting should work IF the library uses absolute angles over the physical servo range.
        angle_for_servo_lib = clamped_angle # If actuation_range in Servo() was the total physical range (e.g. 180)
                                            # then this value should be correct.
        
        try:
            servo_obj.angle = angle_for_servo_lib
            self.SERVO_PARAMS[servo_name]['current_angle'] = clamped_angle
            if not initial_setup:
                print(f"Servo '{servo_name}': set to {clamped_angle:.1f}° (lib val: {angle_for_servo_lib:.1f})")
        except Exception as e:
            print(f"Error setting angle for servo '{servo_name}': {e}")

    def go_to_home_position(self):
        if not self.pca: print("Arm not ready (PCA)."); return
        print("Arm moving to home position...")
        # Move in a specific order for less collision risk (e.g. lift elbow/shoulder first)
        order = ['elbow', 'shoulder', 'wrist_ud', 'wrist_rot', 'base', 'claw'] 
        for name in order:
            if name in self.SERVO_PARAMS:
                self.set_servo_angle(name, self.SERVO_PARAMS[name]['default_angle'])
                time.sleep(0.15)
        print("Arm at home position.")

    def operate_gripper(self, state, position=None): # state "open", "close", or "set"
        if not self.pca: print("Arm not ready (PCA)."); return
        claw_params = self.SERVO_PARAMS['claw']
        open_val = claw_params['min_angle']   # Example: you calibrate this
        closed_val = claw_params['max_angle'] # Example: you calibrate this

        if position is not None:
            target_angle = position
            print(f"Gripper: Setting to {target_angle}°")
        elif state == "open":
            target_angle = open_val
            print("Gripper: Opening")
        elif state == "close":
            target_angle = closed_val
            print("Gripper: Closing")
        else:
            print(f"Gripper: Unknown state '{state}'."); return
        self.set_servo_angle('claw', target_angle)

    def test_servo_increment(self, servo_name, increment=5):
        if not self.pca: print("Arm not ready (PCA)."); return
        if servo_name not in self.SERVO_PARAMS: print(f"Unknown servo: {servo_name}"); return
        current = self.SERVO_PARAMS[servo_name]['current_angle']
        self.set_servo_angle(servo_name, current + increment)
        
    def test_servo_decrement(self, servo_name, decrement=5):
        if not self.pca: print("Arm not ready (PCA)."); return
        if servo_name not in self.SERVO_PARAMS: print(f"Unknown servo: {servo_name}"); return
        current = self.SERVO_PARAMS[servo_name]['current_angle']
        self.set_servo_angle(servo_name, current - decrement)

    def print_servo_angles(self):
        if not self.pca: print("Arm not ready (PCA)."); return
        print("\n--- Current Servo Angles ---")
        for name, params in self.SERVO_PARAMS.items():
            print(f"Servo '{name}': {params['current_angle']:.1f}° (Min: {params['min_angle']}, Max: {params['max_angle']}, Default: {params['default_angle']})")
        print("---------------------------\n")
        print("To save: Manually update SERVO_PARAMS in the script with these values if they are desired defaults or limits.")

    def move_piece_on_board(self, from_sq_uci, to_sq_uci, captured_piece=False):
        if not self.pca: print("Arm not ready (PCA)."); return
        print(f"!!! ARM ACTION (Placeholder): Would move piece from {from_sq_uci} to {to_sq_uci} !!!")
        print("    This requires: Board-to-Arm Coordinate Transformation & Inverse Kinematics.")
        # Example sequence (highly conceptual)
        # 1. Calculate board square physical XYZ coordinates relative to arm base
        #    (e.g., x_from, y_from = self.get_board_xy(from_sq_uci))
        # 2. Go above 'from' square (Z_hover)
        #    (e.g., self.move_to_xyz(x_from, y_from, Z_HOVER))
        # 3. Open gripper: self.operate_gripper("open")
        # 4. Lower to 'from' square (Z_pick): self.move_to_xyz(x_from, y_from, Z_PICK)
        # 5. Close gripper: self.operate_gripper("close"); time.sleep(0.5)
        # 6. Lift piece (Z_hover): self.move_to_xyz(x_from, y_from, Z_HOVER_WITH_PIECE)
        # 7. Go above 'to' square (Z_hover)
        # 8. Lower to 'to' square (Z_place)
        # 9. Open gripper: self.operate_gripper("open"); time.sleep(0.5)
        # 10. Lift arm (Z_hover)
        # 11. self.go_to_home_position()
        time.sleep(1) # Simulate some time for the placeholder

    def release_servos(self):
        if self.pca:
            print("Releasing all servos (setting duty_cycle to 0)...")
            for i in range(16): # PCA9685 has 16 channels
                try: self.pca.channels[i].duty_cycle = 0
                except: pass # Ignore if channel not used as servo
            print("Servos released. They might be movable by hand now.")
        else: print("PCA not init, cannot release servos.")

# --- Initialize Webcam ---
CAM_INDEX = 0 
cap = None
try:
    cap = cv2.VideoCapture(CAM_INDEX) 
    if not cap.isOpened():
        print(f"INFO: Webcam at index {CAM_INDEX} failed, trying index 1...")
        CAM_INDEX = 1; cap = cv2.VideoCapture(CAM_INDEX)
        if not cap.isOpened(): raise IOError(f"Cannot open webcam. Tried 0 and 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"Webcam OK (index {CAM_INDEX}, W: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, H: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}).")
except Exception as e: print(f"CRITICAL Error initializing webcam: {e}"); exit()

# --- Load the chess engine ---
engine_path = "/usr/games/stockfish" # Default for Debian/Pi OS systems
engine = None
try:
    if not os.path.exists(engine_path) or not os.access(engine_path, os.X_OK):
        raise FileNotFoundError(f"Stockfish not found or not executable at '{engine_path}'. Try: sudo apt install stockfish")
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    # engine.configure({"Threads": 2, "Hash": 64}) # Optional: Tune for Pi
    print(f"Stockfish engine OK: {engine_path}")
except Exception as e:
    print(f"Error loading Stockfish: {e}"); engine = None

# --- Chess game state & Utility Functions (Keep from previous versions) ---
current_chess_board = chess.Board()
def segment_board(img): # Unchanged (use your existing compact version)
    if img is None or img.shape[0] == 0 or img.shape[1] == 0: return []
    height, width = img.shape[:2]; square_h, square_w = height // 8, width // 8; squares = []
    for r in range(8):
        for c_ in range(8): y1,y2=r*square_h,(r+1)*square_h; x1,x2=c_*square_w,(c_+1)*square_w; squares.append(((r,c_),img[y1:y2,x1:x2]))
    return squares
def highlight_pawn_positions(img, b): # Unchanged (use your existing compact version)
    if img is None or img.shape[0]==0: return img
    sh,sw=img.shape[0]//8,img.shape[1]//8; hi=img.copy()
    for sq_idx in chess.SQUARES:
        p=b.piece_at(sq_idx)
        if p and p.piece_type==chess.PAWN: r,c=7-chess.square_rank(sq_idx),chess.square_file(sq_idx);tl,br=(c*sw,r*sh),((c+1)*sw,(r+1)*sh);hi=cv2.rectangle(hi,tl,br,(0,255,0)if p.color else(0,0,255),2)
    return hi
def get_user_move(prev,new): return None # Unchanged

# --- Main Loop ---
def main():
    global current_chess_board
    warped_board_output_size = 480 
    board_detector = BoardDetector(output_size=warped_board_output_size)
    arm_controller = ArmController()

    RUN_IN_DEBUG_MODE = False # SET True FOR BOARD DETECTION TUNING
    print("\n--- Chess Vision Arm Control (Raspberry Pi 4) ---")
    print("Board Vision: 'q'=Quit | Game: 'm'=Engine Move, 'u'=Undo, 'r'=Reset")
    print("Arm Control: 'h'=Home, 'o'=Open Claw, 'c'=Close Claw, 'l'=Release Servos")
    print("Arm Calibrate: '1-6' Select Servo | '+' Increment | '-' Decrement | 'p' Print Angles")
    
    active_test_servo = 'base' 
    engine_play_requested = False

    if arm_controller.pca is None:
        print("CRITICAL WARNING: ARM CONTROLLER FAILED TO INITIALIZE. ARM FUNCTIONS DISABLED.")

    while True:
        ret, frame = cap.read()
        if not ret: print("ERR: Frame skip."); time.sleep(0.1); continue
        
        processed_frame = frame 
        warped_board_img, board_corners = board_detector.detect(processed_frame.copy(), debug_mode=RUN_IN_DEBUG_MODE)
        display_frame = processed_frame.copy()

        # Display logic for board... (condensed for brevity, same as before)
        if warped_board_img is not None:
            cv2.imshow("1. Warped Board", warped_board_img)
            #highlighted = highlight_pawn_positions(warped_board_img, current_chess_board) # Can be slow
            #cv2.imshow("2. Warped+Pawns", highlighted)
            if board_corners is not None: cv2.polylines(display_frame,[np.int32(board_corners)],True,(0,255,0),2,cv2.LINE_AA)
        else:
            dummy=np.zeros((warped_board_output_size,warped_board_output_size,3),dtype=np.uint8)
            cv2.putText(dummy,"No Board",(20,warped_board_output_size//2-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            cv2.imshow("1. Warped Board",dummy); #cv2.imshow("2. Warped+Pawns",dummy)

        cv2.putText(display_frame,f"TestServo:{active_test_servo.upper()}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),1,cv2.LINE_AA)
        cv2.putText(display_frame,"Keys: m,u,r,q | h,o,c,l | 1-6,+,-,p",(10,display_frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,50),1,cv2.LINE_AA)
        cv2.imshow("3. Live Feed", display_frame)
        key = cv2.waitKey(1) & 0xFF

        # Engine move execution block
        if engine_play_requested and arm_controller.pca:
            engine_play_requested = False # consume request
            if engine and not current_chess_board.is_game_over():
                try:
                    print("Engine thinking for physical move..."); time.sleep(0.1) # allow print to show
                    limit = chess.engine.Limit(time=0.2, depth=5) # Quick analysis for Pi
                    result = engine.play(current_chess_board, limit)
                    if result.move:
                        print(f"Engine wants to play: {result.move.uci()}")
                        # Physical arm movement call
                        # Note: UCI format for move_piece_on_board
                        arm_controller.move_piece_on_board(result.move.uci()[:2], result.move.uci()[2:], current_chess_board.is_capture(result.move))
                        current_chess_board.push(result.move) # Update internal board state *after* successful physical move
                        print(current_chess_board)
                    else: print("Engine couldn't find move.")
                except Exception as e: print(f"Engine/Arm play error: {e}")
            elif not engine: print("Engine not loaded for play.")
            elif current_chess_board.is_game_over(): print("Game over, engine won't play.")


        # Key press handling
        if key == ord('m'):
            if arm_controller.pca and engine: engine_play_requested = True; print("Engine move requested.")
            else: print("Engine or Arm not ready for 'm'.")
        elif key == ord('u'): # Undo
            if current_chess_board.move_stack:current_chess_board.pop(); # Pop engine
            if current_chess_board.move_stack:current_chess_board.pop(); # Pop human (if applicable)
            print("Undo.");print(current_chess_board)
        elif key == ord('r'): # Reset
            current_chess_board.reset();board_detector.last_known_corners=None;board_detector.consecutive_misses=0
            if arm_controller.pca:arm_controller.go_to_home_position()
            print("Board & Arm Reset.");print(current_chess_board)
        elif key == ord('h') and arm_controller.pca: arm_controller.go_to_home_position()
        elif key == ord('o') and arm_controller.pca: arm_controller.operate_gripper("open")
        elif key == ord('c') and arm_controller.pca: arm_controller.operate_gripper("close")
        elif key == ord('l') and arm_controller.pca: arm_controller.release_servos()
        elif key == ord('p') and arm_controller.pca: arm_controller.print_servo_angles()
        elif ord('1') <= key <= ord('6'):
            servo_map = {ord('1'):'base', ord('2'):'shoulder', ord('3'):'elbow', ord('4'):'wrist_ud', ord('5'):'wrist_rot', ord('6'):'claw'}
            active_test_servo = servo_map[key]; print(f"Selected Servo: {active_test_servo.upper()}")
        elif key == ord('+') or key == ord('='):
            if arm_controller.pca: arm_controller.test_servo_increment(active_test_servo, 5) # 5 degree increment
        elif key == ord('-'):
            if arm_controller.pca: arm_controller.test_servo_decrement(active_test_servo, 5) # 5 degree decrement
        elif key == ord('q'): print("Quitting."); break
        
        if current_chess_board.is_game_over() and not RUN_IN_DEBUG_MODE: print("Game Over."); time.sleep(1); break
    
    print("Cleaning up...")
    if arm_controller.pca: arm_controller.release_servos()
    if engine:
        try:
            engine.quit()
        except:
            pass
    if cap: cap.release()
    cv2.destroyAllWindows()
    # Final game state messages... (as before)

if __name__ == "__main__":
    main()