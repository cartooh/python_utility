import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
#os.environ["OPENCV_LOG_LEVEL"] = "DEBUG"
#os.environ["OPENCV_VIDEOIO_DEBUG"] = "1"

import cv2



# ffmpeg -list_devices true -f dshow -i dummy
# ffmpeg -f dshow -list_options true -i video="HD Pro Webcam C920"
# ffmpeg -f dshow -video_size 1920x1080 -vcodec h264 -i video="HD Pro Webcam C920" test.mp4
def frame_reader_av(args):
    import logging
    import av

    logging.basicConfig()
    logging.getLogger('libav').setLevel(logging.FATAL)
    container = av.open(**args)
    try:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format='bgr24')
    finally:
        print("finally close")
        container.close()


def frame_reader_cv(param=dict(), **kwargs):
    param.update(kwargs)
    args = dict(index=param['index'])
    if 'apiPreference' in param:
        args['apiPreference'] = param['apiPreference']
    
    try:
        print(args)
        cap = cv2.VideoCapture(**args)
        if 'width' in param:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, param['width'])
        if 'height' in param:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, param['height'])
        while True:
            _, frame = cap.read()
            yield frame
    finally:
        print("finally release")
        cap.release()


def frame_reader_async(reader):
    import threading
    import queue

    _cancel = threading.Event()
    _queue = queue.Queue(maxsize=32)
    
    def _reader_mt():
        try:
            for i, frame in enumerate(reader):
                if _queue.full():
                    # キューがいっぱいの場合、古いキューを破棄して、新しいフレームをputする
                    # 画像処理で遅れている場合に、リアルタイム性を優先し、新しい画像が反映されやすくする
                    _queue.get_nowait()
                _queue.put((i, frame))

                if _cancel.is_set():
                    break
        finally:
            _cancel.set()
            print("cancel.set()")
    
    th = threading.Thread(target=_reader_mt)
    th.start()
    
    try:
        while True:
            try:
                _, frame = _queue_frame.get(timeout=1.0)
                yield frame
                
            except queue.Empty:
                if _cancel.is_set():
                    break
    finally:
        _cancel.set()
        th.join()
        print("th.join()")


def get_frame_size(reader):
    for frame in reader:
        return frame.shape



def show_frame(frame_reader, 
    dirname="capture",
    fourcc_str='mp4v', fps=30,
    image_filename='{dirname}/image_{dt}.png',
    video_filename='{dirname}/video_{dt}.avi',
    dt_format='%Y%m%d_%H%M%S_%f',
    display=None, keyboard=None):
    
    if display is None:
        def _display(frame):
            cv2.imshow("capture", frame)
        display = _display
    
    if keyboard is None:
        def _keyboard(key):
            pass
        keyboard = _keyboard
    
    from os import makedirs
    from datetime import datetime
    import time 
    
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    makedirs(dirname, exist_ok=True)
    video = None
    
    try:
        for frame in frame_reader:
            display(frame)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord("r") and not is_record:
                dt = datetime.now().strftime(dt_format)
                fn = video_filename.format(dirname=dirname, dt=dt)
                
                start_time = time.time()
                h, w = frame.shape[:2]
                
                print("record", fn, w, h, fps)
                video = cv2.VideoWriter(fn, fourcc, fps, (w, h))
                
            if video:
                count += 1
                video.write(frame)

            if key == ord("s") and video:
                stop_time = time.time()

                print("stop", stop_time - start_time, count, count / (stop_time - start_time))
                video.release()
                video = None
                continue

            if key == ord('c'):
                dt = datetime.now().strftime(dt_format)
                fn = image_filename.format(dirname=dirname, dt=dt)
                print("capture", fn)
                cv2.imwrite(fn, frame1)
                continue
            
            if key == 27:  # ESC
                break
            
            if key == 0xff:
                continue
            
            keyboard(key)

    finally:
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
    show_frame(frame_reader_cv(index=1, apiPreference=cv2.CAP_MSMF), keyboard=lambda key: print(key))
    