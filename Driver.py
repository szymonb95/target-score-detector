from VideoAnalyzer import VideoAnalyzer
from Sketcher import Sketcher
import cv2

# input
# model = cv2.imread('res/input/target.jpg')
# video_name = 'res/input/video.mp4'
model = cv2.imread('res/input/target_hq.jpg')
model = cv2.resize(model, (300, 300))
video_name = 'res/input/02.mp4'
bullseye_point = (355, 370)
inner_diameter_px = 30
inner_diameter_cm = 1.5
rings_amount = 10
display_in_cm = True

# get a sample frame from the video
# cap = cv2.VideoCapture(video_name)
# camera capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
_, test_sample = cap.read()

# calculate the sizes of the frame and the input
model_h, model_w, _ = model.shape
frame_h, frame_w, _ = test_sample.shape
pixel_to_cm = inner_diameter_cm / inner_diameter_px
pixel_to_inch = pixel_to_cm * 2.54
measure_unit = pixel_to_cm if display_in_cm else pixel_to_inch
measure_unit_name = 'cm' if display_in_cm else '"'

# analyze
sketcher = Sketcher(measure_unit, measure_unit_name)
video_analyzer = VideoAnalyzer(video_name, model, bullseye_point, rings_amount, inner_diameter_px)
video_analyzer.analyze('res/output/output.mp4', sketcher)