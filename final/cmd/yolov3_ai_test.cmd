rem darknet.exe detector test test/obj.data test/yolov3_ai.cfg test/yolov3_ai_last.weights test/1.jpg -thresh 0.2
darknet.exe detector demo test/obj.data test/yolov3_ai.cfg test/yolov3_ai_last.weights data/test_video.mp4 -thresh 0.25 -i 0 -out_filename play.avi
pause