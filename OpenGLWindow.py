from __future__ import annotations
from abc import ABC, abstractmethod
from threading import Thread, Event
from queue import Queue
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.WGL import wglGetCurrentDC
import cv2
import numpy as np
import win32gui
import win32con
import win32com.client
import sys
from screeninfo import get_monitors
from collections import deque
import time

class Command(ABC):
    """
    The Command interface declares a method for executing a command.
    """

    @abstractmethod
    def execute(self) -> None:
        pass


class QuitCommand(Command):
    def execute(self) -> None:
        pass


class CallbackCommand(Command):
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
    def execute(self):
        self.func(*self.args, **self.kwargs)


class FutureCommand(Command):
    class TimeOut(RuntimeError):
        pass
    
    
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.__event = Event()
        self.__result = None
    
    def execute(self):
        self.__result = self.func(*self.args, **self.kwargs)
        self.__event.set()
    
    def get(self, timeout=None):
        if self.__event.wait(timeout=timeout):
            return self.__result
        else:
            raise self.TimeOut()

        


class Worker:
    def __init__(self, daemon=False, finalize=None):
        self.queue = Queue()
        self.is_done = False
        self.thread = Thread(target=self.run, daemon=daemon)
        self.finalize = finalize
    
    def run(self):
        try:
            while True:
                item = self.queue.get()
                if isinstance(item, QuitCommand):
                    break
                item.execute()
        finally:
            self.is_done = True
            if self.finalize is not None:
                self.finalize()
        
    
    def start(self):
        self.thread.start()
    
    def is_alive(self):
        return self.thread.is_alive()
        
    def join(self):
        self.thread.join()
    
    def put(self, cmd):
        self.queue.put(cmd)
    
    def putQuitCommand(self):
        self.put(QuitCommand())
    
    def putCallback(self, func, *args, **kwargs):
        self.put(CallbackCommand(func, *args, **kwargs))
        
    def putFuture(self, func, *args, **kwargs):
        future = FutureCommand(func, *args, **kwargs)
        self.put(future)
        return future
        

class CalcFPS:
    def __init__(self, n=10):
        self.reset(n)
    
    def reset(self, n=10):
        # 直近N_SAMPLE分の時刻をキューで保持。FPSの計算用
        self.n = n
        self.q = deque([time.time() for i in range(n)])
    
    def __call__(self):
        # 直近N_SAMPLEフレームからfps算出
        now = time.time()
        fps = self.n / (now - self.q.popleft() + 0.0000001)
        self.q.append(now)
        return fps
        
        


class OpenGLWindow:
    @classmethod
    def __get_manager(cls):
        return cls.Manager.get_instance()
    
    @classmethod
    def __reset_manager(cls):
        delattr(cls.Manager, "_instance")
    
    
    @classmethod
    def __putCallback(cls, func, *args, **kwargs):
        cls.__get_manager().putCallback(func, *args, **kwargs)
    
    
    @classmethod
    def __putFuture(cls, func, *args, **kwargs):
        return cls.__get_manager().putFuture(func, *args, **kwargs)
    
    
    @classmethod
    def start(cls):
        cls.__get_manager().start()


    @classmethod
    def join(cls):
        cls.__get_manager().join()


    @classmethod
    def is_alive(cls):
        return cls.__get_manager().is_alive()

        
    @property
    @classmethod
    def terminated(cls):
        return cls.__get_manager().terminated
    
    
    @classmethod
    def terminate(cls):
        print(f'terminate {glutGetWindow()}')
        cls.__get_manager().terminate()
        glutLeaveMainLoop()
        
        
    class Manager:
        @classmethod
        def get_instance(cls):
            if not hasattr(cls, "_instance"):
                cls._instance = cls()
            return cls._instance
        
        
        def __init__(self):
            self.__winids = []
            self.__terminated = False
            self.__worker = Worker(daemon=True)
            self.__worker.start()
        
        
        def append(self, win):
            self.__winids.append(win.winid)
        
        
        def remove(self, win):
            self.__winids.remove(win.winid)
        
        
        @property
        def terminated(self):
            return self.__terminated


        def terminate(self):
            print(f'self.__terminated = True')
            self.__terminated = True
            
        
        def putCallback(self, func, *args, **kwargs):
            self.__worker.putCallback(func, *args, **kwargs)
        
        
        def putFuture(self, func, *args, **kwargs):
            return self.__worker.putFuture(func, *args, **kwargs)
            
            
        def is_alive(self):
            return self.__worker.is_alive() 
        
        
        def join(self):
            self.__worker.join()
        
            
        def __mainloopevent(self):
            for winid in self.__winids:
                glutSetWindow(winid)
                glutPostRedisplay()
                glutMainLoopEvent()
                if self.__terminated:
                    break
            
            if self.__terminated:
                while len(self.__winids) > 0:
                    winid = self.__winids.pop()
                    print("glutDestroyWindow", winid)
                    glutDestroyWindow(winid)
                self.__worker.putQuitCommand()
            else:
                self.putCallback(self.__mainloopevent)
        
        
        def __start(self):
            if len(self.__winids) == 0:
                raise RuntimeError("len(self.__winids) == 0")

            glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS)
            self.putCallback(self.__mainloopevent)
            
        
        def start(self):
            self.putCallback(self.__start)
    
    
    def __update_image(self, image):
        self.__image = image
    
    
    def update_image(self, image):
        self.__putCallback(self.__update_image, image)
    
    
    def __display_def(self):
        # Paste into texture to draw at high speed
        img = self.__image.copy()
        # # FPSの値を描画
        # if not hasattr(self, "__fps"):
        #     self.__fps = CalcFPS(1000)
        # cv2.putText(img,'{:6.3f}fps'.format(self.__fps()), (10,50),
        #             cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #BGR-->RGB
        h, w = img.shape[:2]
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glColor3f(1.0, 1.0, 1.0)
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glColor3f(1.0, 1.0, 1.0)

        # Enable texture map
        glEnable(GL_TEXTURE_2D)
        # Set texture map method
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # draw square
        glBegin(GL_QUADS) 
        # 左下
        glTexCoord2d(0.0, 1.0)
        glVertex3d(-w/2, -h/2, 0.0)
        # 右下
        glTexCoord2d(1.0, 1.0)
        glVertex3d( w/2, -h/2,  0.0)
        # 右上
        glTexCoord2d(1.0, 0.0)
        glVertex3d( w/2,  h/2,  0.0)
        # 左上
        glTexCoord2d(0.0, 0.0)
        glVertex3d(-w/2,  h/2,  0.0)
        glEnd()
        
        glFlush();
        glutSwapBuffers()
    
    
    def __reshape_def(self, width, height):
        self.__width  = width
        self.__height = height
        glViewport(0, 0, width, height)
        glLoadIdentity()
        #Make the display area proportional to the size of the view
        if not self.__autoScale:
            glOrtho(-width/2, width/2, -height/2, height/2, -1.0, 1.0)
        elif self.__aspect == 'equal':
            h = max(self.__imageHeight, self.__imageWidth * height / width)
            w = h * width / height
            glOrtho(-w/2, w/2, -h/2, h/2, -1.0, 1.0)
        else:
            glOrtho(-self.__imageWidth/2, self.__imageWidth/2, -self.__imageHeight/2, self.__imageHeight/2, -1.0, 1.0)
    
    
    def __init_def(self):
        glClearColor(0.7, 0.7, 0.7, 0.7)
        
        
    def __pre_close_def(self, win):
        return False
    
    
    def __close_def(self):
        self.__get_manager().remove(self)
        self.__pre_close(self)
        self.terminate()

    
    def __pre_keyboard_def(self, win, key, x, y):
        return False
    
    
    def __keyboard_def(self, key, x, y):
        if self.__pre_keyboard(self, key, x, y):
            return
        
        # convert byte to str
        key = key.decode('utf-8')
        # press q to exit
        if key == 'q':
            self.terminate()
        elif key == 'f':
            self.toggle_fullscreen()
    
    
    def __mouse_def(self, button, state, x, y):
        pass
        
    
    def __getWindowRect(self):
        return win32gui.GetWindowRect(self.__hWnd)
    
    
    def getWindowRectAsync(self):
        return self.__putFuture(self.__getWindowRect)
    
    
    def getWindowRectSync(self):
        return self.getWindowRectAsync().get()
    
    
    def __showWindow(self, nCmdShow):
        # print(sys._getframe().f_code.co_name, self.__hWnd, nCmdShow)
        win32gui.ShowWindow(self.__hWnd, nCmdShow)
    
    
    def maximize(self):
        # print(sys._getframe().f_code.co_name, self.__hWnd)
        self.__putCallback(self.__showWindow, win32con.SW_MAXIMIZE)
    
    
    def restore(self):
        # print(sys._getframe().f_code.co_name, self.__hWnd)
        self.__putCallback(self.__showWindow, win32con.SW_RESTORE)
    
    
    def __focus_window(self):
        # print(sys._getframe().f_code.co_name, self.__hWnd)
        shell = win32com.client.Dispatch("WScript.Shell")
        win32gui.ShowWindow(self.__hWnd, win32con.SW_SHOW)
        win32gui.ShowWindow(self.__hWnd, win32con.SW_SHOWNOACTIVATE)
        shell.SendKeys('%')
        win32gui.SetForegroundWindow(self.__hWnd)
        shell.SendKeys('{ESC}')
        
    def focus_window(self):
        # print(sys._getframe().f_code.co_name, self.__hWnd)
        self.__putCallback(self.__focus_window)


    def __show_titlebar(self):
        # print(sys._getframe().f_code.co_name, self.__hWnd)
        style = win32gui.GetWindowLong(self.__hWnd, win32con.GWL_STYLE)
        style |= win32con.WS_CAPTION
        win32gui.SetWindowLong(self.__hWnd, win32con.GWL_STYLE, style)
        
        
    def show_titlebar(self):
        # print(sys._getframe().f_code.co_name, self.__hWnd)
        self.__putCallback(self.__show_titlebar)

        
    def __hide_titlebar(self):
        # print(sys._getframe().f_code.co_name, self.__hWnd)
        style = win32gui.GetWindowLong(self.__hWnd, win32con.GWL_STYLE)
        style &= ~win32con.WS_CAPTION
        win32gui.SetWindowLong(self.__hWnd, win32con.GWL_STYLE, style)
        
        
    def hide_titlebar(self):
        # print(sys._getframe().f_code.co_name, self.__hWnd)
        self.__putCallback(self.__hide_titlebar)
    
    
    def toggle_fullscreen(self):
        self.__fullscreen = not self.__fullscreen
        if self.__fullscreen:
            self.hide_titlebar()
            self.maximize()
        else:
            self.show_titlebar()
            self.restore()
        
    
    
    def __createWindow(self):
        glutInitWindowPosition(self.__x, self.__y);
        glutInitWindowSize(self.__width, self.__height);
        glutInit(self.__argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)
        """
        GLUT_RGBA	RGBA モード。GLUT_RGBAも GLUT_INDEX も記述されないときのデフォルト
        GLUT_RGB	GLUT_RGBA と同じ
        GLUT_INDEX	カラーインデックス モード。GLUT_RGBAも記述された場合，上書きする
        
        GLUT_SINGLE	シングルバッファ モード。GLUT_DOUBLE もGLUT_SINGLEも記述されていない場合の デフォルトである
        GLUT_DOUBLE	ダブルバッファ モード。GLUT_SINGLE も記述されていた場合，上書きする
        GLUT_ACCUM	アキュムレーション バッファ
        
        GLUT_ALPHA	カラーバッファにアルファ成分を加えること
        GLUT_DEPTH	デプス(Z)バッファを加えること
        GLUT_STENCIL	ステンシル・バッファを加えること
        GLUT_MULTISAMPLE	マルチサンプリングのサポート。マルチサンプリングが利用できない場合は無視される
        GLUT_STEREO	ステレオ・ウィンドウビットマスク
        """
        
        def wrapper(f):
            def _wrapper(*args, **kwargs):
                try:
                    return f(*args, **kwargs)
                except:
                    import traceback
                    traceback.print_exc()
                    self.terminate()
            return _wrapper
                
        
        self.__winid = glutCreateWindow(self.__title)
        self.__hWnd = win32gui.WindowFromDC(wglGetCurrentDC())
        glutDisplayFunc(wrapper(self.__display))
        glutReshapeFunc(wrapper(self.__reshape))
        glutKeyboardFunc(wrapper(self.__keyboard))
        glutMouseFunc(wrapper(self.__mouse))
        glutCloseFunc(wrapper(self.__close))
        self.__get_manager().append(self)
        self.__init()
        
        if self.__fullscreen:
            self.hide_titlebar()
            self.maximize()
        
    @property
    def winid(self):
        return self.__winid
    

    def __init__(
        self, argv=(), 
        x=0, y=0, width=720, height=480, title="Display", image=None,
        init=None, display=None, reshape=None,
        pre_keyboard=None, keyboard=None, mouse=None, 
        pre_close=None, close=None,
        autoScale=True, aspect='equal', fullscreen=False,
    ):
        self.__init         = init         if init         is not None else self.__init_def
        self.__display      = display      if display      is not None else self.__display_def
        self.__reshape      = reshape      if reshape      is not None else self.__reshape_def
        self.__pre_keyboard = pre_keyboard if pre_keyboard is not None else self.__pre_keyboard_def
        self.__keyboard     = keyboard     if keyboard     is not None else self.__keyboard_def
        self.__mouse        = mouse        if mouse        is not None else self.__mouse_def
        self.__pre_close    = pre_close    if pre_close    is not None else self.__pre_close_def
        self.__close        = close        if close        is not None else self.__close_def
        self.__image        = image        if image        is not None else np.full((height, width, 3), 255, np.uint8)
        
        self.__argv = argv
        self.__x = x
        self.__y = y
        self.__title = title
        self.__width  = width
        self.__height = height
        self.__imageWidth  = width
        self.__imageHeight = height
        self.__autoScale = autoScale
        self.__aspect = aspect
        self.__fullscreen = fullscreen
        self.__hWnd = None
        self.__winid = None
        self.__terminated = False
        
        if not self.is_alive():
            self.__reset_manager()
        
        self.__putCallback(self.__createWindow)
        
        

if __name__ == "__main__":
    g_angle = 0;
    def scene():
        global g_angle

        glPushMatrix();
        glRotated(g_angle, 0, 1, 0);
        glutSolidTeapot(1.0);
        glPopMatrix();
        
        glPushMatrix()
        glColor3fv((1, 0, 0))
        glWindowPos2f(100, 100)
        # glTranslatef(0, 20, 0)
        # glRasterPos3d(0, 0, 0) #//0,0,0位置をスタート位置にする
        
        if not hasattr(scene, "__fps"):
            scene.__fps = CalcFPS()
        
        glutBitmapString(GLUT_BITMAP_HELVETICA_18, f"{scene.__fps()}".encode('ascii'));        
        glPopMatrix()

        g_angle += 1;

    
    def display():
        # print(f"display: {glutGetWindow()}")
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_MODELVIEW) # モデルビュー変換行列の設定
        glLoadIdentity()           # 投影変換の変換行列を単位行列で初期化

        scene();

        glutSwapBuffers();
        
    def reshape(w, h):
        # print(f"reshape: {glutGetWindow()}")
        glViewport(0, 0, w, h);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(40.0, w/h, 0.1, 100.0);

        gluLookAt(0, 2, 5, 0, 0, 0, 0, 1, 0);

    def init():
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glEnable(GL_DEPTH_TEST);

        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
    
    
   
    monitors = get_monitors()
    m = monitors[0]
    win_teapot = OpenGLWindow(title="Teapot", display=display, reshape=reshape, init=init, x=m.x, y=m.y)
    if len(monitors) > 1:
        m = monitors[1]
    #win_capture = OpenGLWindow(title=f"Capture", x=m.x, y=m.y, width=m.width, height=m.height)
    win_capture = OpenGLWindow(title=f"Capture", x=m.x, y=m.y, width=1920, height=1080)
    wins = [win_teapot, win_capture]
    OpenGLWindow.start()
        
    
    def cv_image_reader(cap):
        try:
            while True:
                _, img = cap.read()
                yield img
        finally:
            print("release")
            cap.release()
    
    def get_cv_reader():
        import cv2
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if cap.isOpened() is False:
            raise("IO Error")
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cv_image_reader(cap)
    
    def av_image_reader(args):
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
    
    def get_av_reader():
        reader_args = dict(
            format='dshow', 
            #file='video=HD Pro Webcam C920',
            file='video=Logi C615 HD WebCam',
            options=dict(
                video_size='1920x1080',
                # vcodec='h264',
                vcodec='mjpeg',
                framerate='30',
                rtbufsize='128'
            ),
        )
        return av_image_reader(reader_args)
    
    def updater(reader):
        if win_capture is None:
            return
        
        fps = CalcFPS()
        for img in reader:
            if win_capture.terminated:
                break
            
            if img is not None:
                # FPSの値を描画
                cv2.putText(img,'{:6.3f}fps'.format(fps()), (10,20),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0))
                win_capture.update_image(img)
    
    th = Thread(target=updater, args=(get_av_reader(),), daemon=True)
    th.start()
    
    
    for w in wins:
        w.focus_window()
        w.toggle_fullscreen()
        print("Rect", w.winid, w.getWindowRectSync())
    
    
    time.sleep(3)
    for w in wins:
        w.toggle_fullscreen()
    
    time.sleep(3)
    for w in wins:
        w.toggle_fullscreen()
        
    while OpenGLWindow.is_alive():
        time.sleep(1)
