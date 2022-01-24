import cv2
from client import Trion_grpc_infer_OD
import os
import datetime, time
import json

saved_folder = "data/fti_cameras/"

token = '&t=' + 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwczpcL1wvMTAuMTAuMzcuNTI6ODg4OFwvdjFcL2F1dGhcL2xvZ2luIiwiaWF0IjoxNjM5OTgyOTczLCJleHAiOjE2NDA1ODc3NzMsIm5iZiI6MTYzOTk4Mjk3MywianRpIjoidXNNc09zSWtvR2poZ3FLdSIsInN1YiI6MTg1MCwicHJ2IjoiMTNlOGQwMjhiMzkxZjNiN2I2M2YyMTkzM2RiYWQ0NThmZjIxMDcyZSIsInRlbmFudCI6Mzh9.XKlelQW4_tYb9BPw_Ng7zqITn1TToRbvuYdaZVcbLfc'

camera_src = ['https://fti.giamsat247.vn/w/live/80/fti-hn-idc-t1.stream/playlist.m3u8?d=13584&x=38',
                'https://fti.giamsat247.vn/w/live/80/fti-hn-cgy.stream/playlist.m3u8?d=13578&x=38',
                'https://fti.giamsat247.vn/w/live/99/fti_dn_cam01_full.stream/playlist.m3u8?d=13497&x=38',
                'https://fti.giamsat247.vn/w/live/107/ftilau3cam9.stream/playlist.m3u8?d=13491&x=38',
                'https://fti.giamsat247.vn/w/live/80/ftilau3cam8.stream/playlist.m3u8?d=13488&x=38',
                'https://fti.giamsat247.vn/w/live/107/ftilau3cam7.stream/playlist.m3u8?d=13485&x=38',
                'https://fti.giamsat247.vn/w/live/80/ftilau3cam6.stream/playlist.m3u8?d=13482&x=38',
                'https://fti.giamsat247.vn/w/live/99/ftilau3cam5.stream/playlist.m3u8?d=13479&x=38',
                'https://fti.giamsat247.vn/w/live/80/ftilau3cam4.stream/playlist.m3u8?d=13476&x=38',
                'https://fti.giamsat247.vn/w/live/107/ftilau3cam3.stream/playlist.m3u8?d=13473&x=38',
                'https://fti.giamsat247.vn/w/live/80/ftilau3cam2.stream/playlist.m3u8?d=13470&x=38',
                'https://fti.giamsat247.vn/w/live/80/ftilau2cam4.stream/playlist.m3u8?d=13464&x=38',
                'https://fti.giamsat247.vn/w/live/80/ftilau2cam2.stream/playlist.m3u8?d=13458&x=38',
                'https://fti.giamsat247.vn/w/live/107/ftilau2cam1.stream/playlist.m3u8?d=13455&x=38',
                'https://fti.giamsat247.vn/w/live/107/fti-lau1-cus.stream/playlist.m3u8?d=13452&x=38',
                'https://fti.giamsat247.vn/w/live/107/fti-lau1-cam10.stream/playlist.m3u8?d=13449&x=38',
                'https://fti.giamsat247.vn/w/live/107/fti-lau1-cam09.stream/playlist.m3u8?d=13446&x=38',
                'https://fti.giamsat247.vn/w/live/107/fti-lau1-cam08.stream/playlist.m3u8?d=13443&x=38',
                'https://fti.giamsat247.vn/w/live/107/fti-lau1-cam07.stream/playlist.m3u8?d=13440&x=38',
                'https://fti.giamsat247.vn/w/live/107/fti-lau1-cam06.stream/playlist.m3u8?d=13437&x=38',
                'https://fti.giamsat247.vn/w/live/107/fti-lau1-cam05.stream/playlist.m3u8?d=13434&x=38',
                'https://fti.giamsat247.vn/w/live/107/fti-lau1-cam04.stream/playlist.m3u8?d=13431&x=38',
                'https://fti.giamsat247.vn/w/live/107/fti-lau01-voice02.stream/playlist.m3u8?d=13428&x=38',
                'https://fti.giamsat247.vn/w/live/107/fti-lau01-voice01.stream/playlist.m3u8?d=13425&x=38'
             ]
print("[INFO] Starting...")
capture_list = [cv2.VideoCapture(src + token) for src in camera_src]
print("[INFO] Number cameras: " + str(len(capture_list)))

### Create triton client
# client = Trion_grpc_infer_OD(url='10.10.37.119:8221', confidence=0.3,)
client = Trion_grpc_infer_OD(url='localhost:8221', confidence=0.3,)

images_dir = saved_folder + '/images'
results_dir = saved_folder + '/results'
if not os.path.exists(images_dir):
    os.mkdir(images_dir)

if not os.path.exists(results_dir):
    os.mkdir(results_dir)


### Create save folders
for i in range(len(camera_src)):
    image_dir = saved_folder + '/images' + '/cam_' + str(i)
    result_dir = saved_folder + '/results' + '/cam_' + str(i)

    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    
        
start_hour = 9
stop_hour = 17
cap_time = 10

now = datetime.datetime.now()
start_cap = now.minute

while True:
    now = datetime.datetime.now()
    weekday = datetime.datetime.today().weekday()
    frames = []
    for i in range(len(capture_list)):
        ret, frame = capture_list[i].read()
        frames.append(frame)
        
    

    if (weekday >= 0 and weekday <=4):
        if (now.hour < 9 or now.hour > 17):
            cap_time = cap_time*2
        next_cap = now.minute
        if abs(next_cap-((start_cap+cap_time)%60)) == 0:
            start_cap = next_cap
            
            print('[INFO] ' + now.strftime("%Y-%m-%d %H-%M-%S") + ' Running...')
            for i in range(len(camera_src)):
                try:
                    status, num_box, classes, score, boxes = client.do_inference_sync(frames[i])

                    ### Draw prediction
        #             output = client.plot_bboxes(frame, num_box, classes, score, boxes)

    #                 cv2.imshow("Capturing", frame)
                    date = now.strftime("%Y_%m_%d_%H_%M_%S")
                    image_file = images_dir + '/cam_' + str(i) + '/frame_' + date + '.jpg'
                    cv2.imwrite(image_file, frames[i])

                    result_file = results_dir + '/cam_' + str(i) + '/frame_' + date + '.json'
                    results = {
                        "num_box": num_box,
                        "classes": classes.tolist(),
                        "score": score.tolist(),
                        "boxes": boxes.tolist(),
                    }
                    with open(result_file, 'w') as f:
                        json.dump(results, f)
                except:
                    print("[ERROR] Problem with ", camera_src[i])


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in capture_list:
    cap.release()
cv2.destroyAllWindows()
